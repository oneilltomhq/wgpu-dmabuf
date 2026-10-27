//! Fencing across devices and processes with `sync_file`.
//!
//! Importing a dma-buf establishes *sharing*. It establishes no *ordering*
//! whatsoever, and the failure mode of missing ordering is not a crash — it is
//! a frame late, every frame, forever. This module is the explicit-sync half.
//!
//! Four mechanisms, all `sync_file`-shaped, all measured on Mesa `anv` /
//! Intel Iris Xe / kernel 7.1:
//!
//! * [`CompletionFence`] — **egress**. "When the next submit on this queue
//!   completes, hand me a `sync_file` fd." The fd another device waits on, or
//!   KMS takes as `IN_FENCE_FD`.
//! * [`AcquireSemaphore`] — **ingress**. Turn someone else's `sync_file` fd
//!   into a wait on the *next* `vkQueueSubmit` wgpu performs on a queue. From
//!   wgpu's point of view nothing changed: the submit it was going to make
//!   anyway gains one entry in `pWaitSemaphores`.
//! * [`export_sync_file`] — the **client** source. Given a dma-buf fd, ask the
//!   kernel (`DMA_BUF_IOCTL_EXPORT_SYNC_FILE`, Linux 5.20+) for the implicit
//!   fences already attached to that buffer. This is how you get an acquire
//!   fence for a producer whose submits you did not issue.
//! * [`import_sync_file`] — the **producer** sink, its mirror
//!   (`DMA_BUF_IOCTL_IMPORT_SYNC_FILE`): attach a [`CompletionFence`]'s
//!   `sync_file` to a dma-buf, so a consumer that syncs implicitly waits for
//!   a render Vulkan would otherwise not have told it about.
//!
//! # What is *not* here
//!
//! **Timeline semaphores.** `wgpu_hal::vulkan::Queue::add_wait_semaphore`
//! accepts `Some(value)` for a timeline payload, and
//! `VK_KHR_external_semaphore_fd` can carry `OPAQUE_FD` timeline handles. This
//! crate implements neither, because nothing in the source material this crate
//! was extracted from ever ran one: `SYNC_FD` is a binary-only handle type, so
//! a timeline export is a different mechanism with different failure modes, and
//! an unproven fence is worse than an absent one.
//!
//! # The extension nobody enables
//!
//! Every semaphore path here needs `VK_KHR_external_semaphore_fd` on the
//! `VkDevice`. **Stock `wgpu::Adapter::request_device` never enables it** —
//! wgpu-hal 30.0.1 does not mention the extension anywhere. Open the device
//! with [`crate::device::open_device`], which adds it by hand and adopts the
//! result through `Adapter::device_from_raw`.

use std::fmt;
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd};
use std::time::Duration;

use ash::vk;

use crate::error::{Error, Result};

use wgpu::hal::api::Vulkan;

// ---------------------------------------------------------------------------
// SyncFile
// ---------------------------------------------------------------------------

/// An owned `sync_file` fd, and the operations that can be performed on one
/// from *outside* the driver that made it.
///
/// `is_signalled`/`wait` use `poll(2)` rather than asking Vulkan whether its own
/// semaphore is ready: grading a fence with the thing being fenced is not
/// grading. A `sync_file` becomes `POLLIN`-readable exactly when its fence
/// signals.
#[derive(Debug)]
pub struct SyncFile {
    fd: OwnedFd,
}

impl SyncFile {
    /// Adopt an existing `sync_file` fd.
    pub fn from_fd(fd: OwnedFd) -> Self {
        Self { fd }
    }

    /// Give up ownership of the fd — e.g. to hand to
    /// [`AcquireSemaphore::import`], which consumes it.
    pub fn into_fd(self) -> OwnedFd {
        self.fd
    }

    /// Borrow the fd.
    pub fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// A second fd naming the same fence.
    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            fd: self.fd.try_clone()?,
        })
    }

    /// Has the fence signalled *right now*? Never blocks.
    pub fn is_signalled(&self) -> Result<bool> {
        Ok(poll(self.fd.as_fd(), Some(Duration::ZERO))?)
    }

    /// Block until the fence signals, or `timeout` elapses. `false` means "not
    /// signalled *yet*".
    ///
    /// There is deliberately no unbounded `wait()`. A fence that never signals
    /// is the failure shape this module exists to make visible, and a harness
    /// that asserts on an unbounded wait cannot report it — it just hangs too.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        Ok(poll(self.fd.as_fd(), Some(timeout))?)
    }

    /// Per-fence provenance. See [`fence_info`].
    pub fn provenance(&self) -> Result<Vec<FenceInfo>> {
        fence_info(self.fd.as_fd())
    }
}

fn poll(fd: BorrowedFd<'_>, timeout: Option<Duration>) -> io::Result<bool> {
    let ms: libc::c_int = match timeout {
        None => -1,
        Some(d) => d.as_millis().min(i32::MAX as u128) as libc::c_int,
    };
    loop {
        let mut pfd = libc::pollfd {
            fd: fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one pollfd, no allocation, fd borrowed for the call.
        let rc = unsafe { libc::poll(&mut pfd, 1, ms) };
        if rc < 0 {
            let e = io::Error::last_os_error();
            // A signal is not an answer about the fence; ask again.
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        return Ok(rc > 0 && (pfd.revents & libc::POLLIN) != 0);
    }
}

// ---------------------------------------------------------------------------
// DMA_BUF_IOCTL_EXPORT_SYNC_FILE
// ---------------------------------------------------------------------------

/// `DMA_BUF_BASE` is `'b'`; `EXPORT_SYNC_FILE` is nr 2, `_IOWR` over an 8-byte
/// struct. Spelled out rather than computed so a reader can check it against
/// `include/uapi/linux/dma-buf.h` without running anything:
///
/// ```text
///   _IOWR('b', 2, struct dma_buf_export_sync_file)
///     = (_IOC_READ|_IOC_WRITE) << 30 | 8 << 16 | 'b' << 8 | 2
/// ```
const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: libc::c_ulong = 0xc008_6202;

/// `struct dma_buf_export_sync_file`. `flags` is in, `fd` is out.
#[repr(C)]
struct ExportSyncFileArg {
    flags: u32,
    fd: i32,
}

/// What the caller is about to do with the buffer — which is what decides which
/// fences the kernel hands back.
///
/// ```text
///   Read  -> the buffer's WRITE fences (someone is writing it)
///   Write -> every fence on the buffer (writes and reads both)
/// ```
///
/// A compositor that samples a client buffer and never writes it wants
/// [`SyncDirection::Read`]. Getting this backwards is silent: `Write` returns
/// *more* fences, so the wait still works and merely over-waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    /// `DMA_BUF_SYNC_READ` — "I am going to read it": the buffer's write fences.
    Read,
    /// `DMA_BUF_SYNC_WRITE` — "I am going to write it": every fence on it.
    Write,
}

impl SyncDirection {
    fn flags(self) -> u32 {
        match self {
            // DMA_BUF_SYNC_READ
            Self::Read => 1 << 0,
            // DMA_BUF_SYNC_WRITE
            Self::Write => 1 << 1,
        }
    }
}

/// One exported `sync_file`, plus the one fact about it that is not visible
/// afterwards.
#[derive(Debug)]
pub struct ExportedFence {
    /// The fence. Feed it to [`AcquireSemaphore::import`], which takes
    /// ownership.
    pub sync_file: SyncFile,
    /// **Was it already signalled the instant the kernel handed it over?**
    ///
    /// `true` means the wait built on it cannot block: either the producer's
    /// work had already retired, or the reservation object was empty and the
    /// kernel substituted an already-signalled stub. Either way this particular
    /// wait ordered nothing, and a run in which *every* export is `true` has
    /// exercised the plumbing without ever exercising the synchronisation.
    ///
    /// Not an error — a producer that finished before it handed the buffer over
    /// is the normal, healthy case. It is a *measurement*, and it is on the
    /// struct rather than in a log line so that a harness can fail on it.
    pub already_signalled: bool,
}

/// Export the implicit fences attached to a dma-buf as a `sync_file`.
///
/// The returned fd is independent of `dmabuf`: closing one does not affect the
/// other.
///
/// # The two traps
///
/// * **It never fails for want of a fence.** If the reservation object is empty
///   the kernel substitutes `dma_fence_get_stub()` — an already-signalled
///   fence — and returns a perfectly valid fd. Hence
///   [`ExportedFence::already_signalled`].
/// * **Neither direction reaches `DMA_RESV_USAGE_BOOKKEEP`**, the class a
///   driver attaches its fences to once userspace has opted into *explicit*
///   synchronisation. For a client on `linux-drm-syncobj-v1` this ioctl still
///   succeeds, still returns a valid fd, and the fd is a signalled stub. So
///   this is the acquire source for **implicit-sync producers**, and it stops
///   being one the moment the producer goes explicit.
///
/// `ENOTTY` means the running kernel predates dma-buf explicit sync (< 5.20).
/// Treat that as fatal rather than silently falling back to implicit sync.
pub fn export_sync_file(dmabuf: BorrowedFd<'_>, direction: SyncDirection) -> Result<ExportedFence> {
    let mut arg = ExportSyncFileArg {
        flags: direction.flags(),
        fd: -1,
    };
    // SAFETY: `arg` is a live `#[repr(C)]` value of exactly the layout the
    // kernel expects, and `dmabuf` is a borrowed fd valid for this call.
    let rc = unsafe {
        libc::ioctl(
            dmabuf.as_raw_fd(),
            DMA_BUF_IOCTL_EXPORT_SYNC_FILE,
            std::ptr::addr_of_mut!(arg),
        )
    };
    if rc != 0 {
        return Err(Error::Io(io::Error::last_os_error()));
    }
    if arg.fd < 0 {
        return Err(Error::Io(io::Error::other(
            "DMA_BUF_IOCTL_EXPORT_SYNC_FILE succeeded but returned no fd",
        )));
    }
    // SAFETY: the kernel just transferred ownership of this fd to us.
    let sync_file = SyncFile::from_fd(unsafe { OwnedFd::from_raw_fd(arg.fd) });
    let already_signalled = sync_file.is_signalled()?;
    Ok(ExportedFence {
        sync_file,
        already_signalled,
    })
}

// ---------------------------------------------------------------------------
// DMA_BUF_IOCTL_IMPORT_SYNC_FILE
// ---------------------------------------------------------------------------

/// `IMPORT_SYNC_FILE` is nr 3, `_IOW` over an 8-byte struct:
///
/// ```text
///   _IOW('b', 3, struct dma_buf_import_sync_file)
///     = _IOC_WRITE << 30 | 8 << 16 | 'b' << 8 | 3
/// ```
///
/// Checked against the kernel's `linux/dma-buf.h` uapi header by compiling it
/// (`0x40086203`, size 8), and recomputed in the tests below.
const DMA_BUF_IOCTL_IMPORT_SYNC_FILE: libc::c_ulong = 0x4008_6203;

/// `struct dma_buf_import_sync_file`. Both fields are in.
#[repr(C)]
struct ImportSyncFileArg {
    flags: u32,
    fd: i32,
}

/// Attach a `sync_file` to a dma-buf's reservation object, so that implicitly
/// synchronised users of the buffer wait for it.
///
/// This is the **producer** half that [`export_sync_file`] is the consumer
/// half of. A Vulkan producer renders into a dma-buf and hands it to someone
/// who syncs implicitly (GL, a media driver, a PipeWire consumer that imports
/// into EGL): Vulkan attached nothing to the buffer, so they would read it
/// before the GPU finished. Export the render's completion
/// ([`CompletionFence`]) and import it here with [`SyncDirection::Write`], and
/// the buffer carries a write fence like a GL render would have left.
///
/// ```text
///   Read  -> added as a read fence: later writers wait for it
///   Write -> added as a write fence: every later user waits for it
/// ```
///
/// The kernel takes its own reference to the fence; `fence` is only borrowed
/// and can be dropped afterwards.
///
/// `ENOTTY` means the kernel predates dma-buf explicit sync (< 5.20); `EINVAL`
/// a bad flag or an fd that is not a `sync_file`. Unlike [`export_sync_file`]
/// there is a sane fallback for a caller here: wait on `fence` on the CPU
/// before handing the buffer on.
pub fn import_sync_file(
    dmabuf: BorrowedFd<'_>,
    fence: BorrowedFd<'_>,
    direction: SyncDirection,
) -> Result<()> {
    let mut arg = ImportSyncFileArg {
        flags: direction.flags(),
        fd: fence.as_raw_fd(),
    };
    // SAFETY: `arg` is a live `#[repr(C)]` value of exactly the layout the
    // kernel expects; both fds are borrowed and valid for this call, and the
    // kernel does not take ownership of either.
    let rc = unsafe {
        libc::ioctl(
            dmabuf.as_raw_fd(),
            DMA_BUF_IOCTL_IMPORT_SYNC_FILE,
            std::ptr::addr_of_mut!(arg),
        )
    };
    if rc != 0 {
        return Err(Error::Io(io::Error::last_os_error()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SYNC_IOC_FILE_INFO
// ---------------------------------------------------------------------------

/// `SYNC_IOC_MAGIC` is `'>'`; `SYNC_IOC_FILE_INFO` is nr 4, `_IOWR` over the
/// 56-byte `struct sync_file_info`:
///
/// ```text
///   _IOWR('>', 4, struct sync_file_info)
///     = (_IOC_READ|_IOC_WRITE) << 30 | 56 << 16 | '>' << 8 | 4
/// ```
const SYNC_IOC_FILE_INFO: libc::c_ulong = 0xc038_3e04;

/// `struct sync_file_info` from `include/uapi/linux/sync_file.h`.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SyncFileInfo {
    name: [u8; 32],
    status: i32,
    flags: u32,
    num_fences: u32,
    pad: u32,
    sync_fence_info: u64,
}

/// `struct sync_fence_info` — one entry per fence inside the `sync_file`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SyncFenceInfoRaw {
    obj_name: [u8; 32],
    driver_name: [u8; 32],
    status: i32,
    flags: u32,
    timestamp_ns: u64,
}

fn cstr32(raw: &[u8; 32]) -> String {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

/// **The provenance of one fence inside a `sync_file`.**
///
/// This is the discriminator [`ExportedFence::already_signalled`] cannot be:
/// that bit reads `true` both for the empty-reservation stub (the hazard) and
/// for a real fence that had simply already retired (the healthy case), and
/// timing cannot separate them.
///
/// **On kernel 7.1 the names are only legible while the fence is unsignalled.**
/// `dma_fence_driver_name` / `dma_fence_timeline_name` return the placeholders
/// `"detached-driver"` / `"signaled-timeline"` for *any* signalled fence
/// without consulting the ops table, because a driver may free everything the
/// ops reach the moment it signals. Measured, same buffer, two readings: a real
/// i915 fence reads `driver="0000:00:02.0" obj="<comm>[<pid>]"` in flight and
/// the placeholders once retired.
///
/// So the useful reading is the **positive** one — *the kernel named something,
/// therefore this is not the stub* — and it is available only on an unsignalled
/// export. A masked name means "unknown", never "real".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FenceInfo {
    /// `dma_fence_ops::get_driver_name` — `"stub"` for the kernel's stub fence.
    pub driver_name: String,
    /// `dma_fence_ops::get_timeline_name` — also `"stub"` for the stub fence.
    pub obj_name: String,
    /// 1 = signalled, 0 = active, < 0 = error.
    pub status: i32,
}

impl FenceInfo {
    /// Is this the kernel's `dma_fence_get_stub()` placeholder — i.e. was the
    /// reservation object empty?
    ///
    /// **Returns `false` for a stub on kernel 7.1** — see the struct docs. A
    /// stub is always signalled and a signalled fence's names are masked before
    /// the ops table is reached, so this cannot fire there. `false` is "not
    /// identified as a stub", never "real".
    ///
    /// Deliberately requires *both* names, because `dma_fence_stub_ops` points
    /// both hooks at the same function.
    pub fn is_stub(&self) -> bool {
        self.driver_name == "stub" && self.obj_name == "stub"
    }
}

/// Read the per-fence provenance out of any `sync_file` fd — one from
/// [`export_sync_file`], or one exported from a `VkSemaphore`.
///
/// An **empty** vector is not an error and is not "no fence": a `sync_file` can
/// report `num_fences == 0` once its fence has been reaped. Treat `[]` as
/// *unknown provenance*, never as "not a stub".
pub fn fence_info(fd: BorrowedFd<'_>) -> Result<Vec<FenceInfo>> {
    let mut hdr = SyncFileInfo::default();
    // SAFETY: `hdr` is a live `#[repr(C)]` value of the layout the kernel
    // expects; `sync_fence_info == 0` is the documented "just count" first pass.
    let rc = unsafe {
        libc::ioctl(
            fd.as_raw_fd(),
            SYNC_IOC_FILE_INFO,
            std::ptr::addr_of_mut!(hdr),
        )
    };
    if rc != 0 {
        return Err(Error::Io(io::Error::last_os_error()));
    }
    let n = hdr.num_fences as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut fences = vec![SyncFenceInfoRaw::default(); n];
    let mut hdr2 = SyncFileInfo {
        num_fences: hdr.num_fences,
        sync_fence_info: fences.as_mut_ptr() as u64,
        ..SyncFileInfo::default()
    };
    // SAFETY: `fences` holds exactly `num_fences` entries of the layout the
    // kernel writes and outlives the call.
    let rc = unsafe {
        libc::ioctl(
            fd.as_raw_fd(),
            SYNC_IOC_FILE_INFO,
            std::ptr::addr_of_mut!(hdr2),
        )
    };
    if rc != 0 {
        return Err(Error::Io(io::Error::last_os_error()));
    }
    Ok(fences
        .iter()
        .take(hdr2.num_fences as usize)
        .map(|f| FenceInfo {
            driver_name: cstr32(&f.driver_name),
            obj_name: cstr32(&f.obj_name),
            status: f.status,
        })
        .collect())
}

/// Render a provenance vector as one line, for a harness to print verbatim.
///
/// `"<none reported>"` is its own string rather than an empty one, because "the
/// kernel listed no fences" and "we never asked" must not look alike in a log.
pub fn provenance_line(fences: &[FenceInfo]) -> String {
    if fences.is_empty() {
        return "<none reported>".to_string();
    }
    fences
        .iter()
        .map(|f| {
            format!(
                "driver={:?} obj={:?} status={}",
                f.driver_name, f.obj_name, f.status
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// The Vulkan side
// ---------------------------------------------------------------------------

/// The `ash` handles this module needs, pulled out of a `wgpu::Device` once.
fn semaphore_ctx(
    device: &wgpu::Device,
) -> Result<(ash::Device, ash::Instance, vk::PhysicalDevice)> {
    // SAFETY: we read handles out of the hal device and destroy nothing wgpu
    // owns. The `ash::Device`/`ash::Instance` clones are dispatch-table copies,
    // not second devices.
    unsafe {
        let hal = device.as_hal::<Vulkan>().ok_or(Error::NotVulkan)?;
        if !hal
            .enabled_device_extensions()
            .contains(&ash::khr::external_semaphore_fd::NAME)
        {
            return Err(Error::NoExternalSemaphoreFd);
        }
        Ok((
            hal.raw_device().clone(),
            hal.shared_instance().raw_instance().clone(),
            hal.raw_physical_device(),
        ))
    }
}

fn external_semaphore_features(
    instance: &ash::Instance,
    pdev: vk::PhysicalDevice,
) -> vk::ExternalSemaphoreFeatureFlags {
    let mut props = vk::ExternalSemaphoreProperties::default();
    let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    // SAFETY: a plain query on the physical device.
    unsafe { instance.get_physical_device_external_semaphore_properties(pdev, &info, &mut props) };
    props.external_semaphore_features
}

/// **Ingress**: a `VkSemaphore` on wgpu's device used as the landing pad for
/// someone else's acquire `sync_file`.
///
/// Created once and re-armed every frame. Import is necessarily *temporary*
/// (`VK_SEMAPHORE_IMPORT_TEMPORARY_BIT` is the only mode `SYNC_FD` semaphores
/// support): the payload is replaced for exactly one wait, after which the
/// semaphore reverts to its permanent unsignalled state. So one object serves
/// for the life of the process instead of leaking a `VkSemaphore` per buffer.
///
/// # Ordering contract
///
/// [`Self::wait_on_next_submit`] stages the wait on the **queue**, not on a
/// command buffer, and the staged wait is consumed by whichever
/// `vkQueueSubmit` happens next on that queue. On a single-threaded frame loop
/// that is deterministic and is exactly what you want. It is **not** safe to
/// stage a wait and then let an unrelated submit slip in front of it: the
/// unrelated submit eats the wait and the work that actually reads the dma-buf
/// runs unguarded. [`Self::cancel_pending_wait`] exists for the error path.
pub struct AcquireSemaphore {
    raw_device: ash::Device,
    ext: ash::khr::external_semaphore_fd::Device,
    semaphore: vk::Semaphore,
}

impl fmt::Debug for AcquireSemaphore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AcquireSemaphore")
            .field("semaphore", &self.semaphore)
            .finish_non_exhaustive()
    }
}

impl AcquireSemaphore {
    /// Mint a binary `VkSemaphore` ready to receive `sync_file` payloads.
    ///
    /// Fails loudly if `VK_KHR_external_semaphore_fd` is not enabled
    /// ([`Error::NoExternalSemaphoreFd`] — see [`crate::device::open_device`])
    /// or if the driver will not import `SYNC_FD` handles. Both mean "explicit
    /// acquire cannot be honoured on this stack", which is fatal, not a cue to
    /// fall back to implicit sync.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let (raw_device, instance, pdev) = semaphore_ctx(device)?;
        // Identity, not plausibility: ask the driver whether a SYNC_FD payload
        // can actually be imported before building a mechanism on the belief.
        if !external_semaphore_features(&instance, pdev)
            .contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE)
        {
            return Err(Error::SyncFdImportUnsupported);
        }
        let ext = ash::khr::external_semaphore_fd::Device::new(&instance, &raw_device);
        // SAFETY: a default create-info on a live device.
        let semaphore =
            unsafe { raw_device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .map_err(Error::vk("vkCreateSemaphore"))?;
        Ok(Self {
            raw_device,
            ext,
            semaphore,
        })
    }

    /// The raw handle, for callers talking to Vulkan themselves.
    pub fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }

    /// Arm this semaphore with a `sync_file`.
    ///
    /// Takes the fence **by value**: `vkImportSemaphoreFdKHR` transfers
    /// ownership of a `SYNC_FD` handle to the driver on success, so the fd must
    /// not be closed by us. On failure it is closed here.
    ///
    /// An already-signalled fence is legal and cheap — the subsequent wait is
    /// satisfied immediately.
    pub fn import(&self, fence: SyncFile) -> Result<()> {
        let raw = fence.into_fd().into_raw_fd();
        let info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(self.semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .fd(raw);
        // SAFETY: `raw` is a live sync_file fd we own until the driver takes it.
        match unsafe { self.ext.import_semaphore_fd(&info) } {
            Ok(()) => Ok(()),
            Err(e) => {
                // The driver did not take the fd; close it rather than leak.
                // SAFETY: still ours, not handed anywhere.
                let _ = unsafe { OwnedFd::from_raw_fd(raw) };
                Err(Error::Vk {
                    what: "vkImportSemaphoreFdKHR",
                    result: e,
                })
            }
        }
    }

    /// Stage this semaphore as a wait on the next `vkQueueSubmit` on `queue`.
    ///
    /// `stage` is where the wait blocks. `FRAGMENT_SHADER` is the honest stage
    /// for "a fragment shader is about to sample this dma-buf";
    /// `ALL_COMMANDS`/`TOP_OF_PIPE` gate the whole submission and are the safe
    /// default when the consumer is a blit or a copy as well.
    ///
    /// Read the type-level ordering contract first.
    pub fn wait_on_next_submit(
        &self,
        queue: &wgpu::Queue,
        stage: vk::PipelineStageFlags,
    ) -> Result<()> {
        // SAFETY: we only stage a semaphore; nothing wgpu owns is destroyed.
        unsafe {
            let hal = queue.as_hal::<Vulkan>().ok_or(Error::NotVulkan)?;
            hal.add_wait_semaphore(self.semaphore, None, stage);
        }
        Ok(())
    }

    /// Un-stage a wait that has not been consumed yet. Returns `true` if it was
    /// still pending.
    ///
    /// The error path: a frame that stages a wait and then decides not to
    /// submit would otherwise leave a live wait for the next frame to trip over.
    pub fn cancel_pending_wait(&self, queue: &wgpu::Queue) -> Result<bool> {
        // SAFETY: as above.
        unsafe {
            let hal = queue.as_hal::<Vulkan>().ok_or(Error::NotVulkan)?;
            Ok(hal.remove_wait_semaphore(self.semaphore))
        }
    }
}

impl Drop for AcquireSemaphore {
    fn drop(&mut self) {
        // SAFETY: created here. Callers are required to have drained the queue
        // (`Device::poll(PollType::wait())`) before dropping this.
        unsafe { self.raw_device.destroy_semaphore(self.semaphore, None) };
    }
}

/// **Egress**: "when the next submit on this queue completes, hand me a
/// `sync_file` fd".
///
/// The mirror image of [`AcquireSemaphore`]. A binary `VkSemaphore` created
/// with `SYNC_FD` as an *exportable* handle type is staged as a signal on the
/// queue ([`Self::arm`]); the submit that follows signals it; exporting
/// ([`Self::export`]) yields a `sync_file` fd that becomes readable when that
/// submission completes.
///
/// A semaphore rather than `VK_KHR_external_fence_fd` because a `VkFence` can
/// only be handed to `vkQueueSubmit` as its `fence` argument, which is wgpu's
/// own submission bookkeeping and not ours to take — whereas the signal
/// semaphore list is already drained into every submit. Same `sync_file` fd out
/// the far end either way, and **zero** changes to `Queue::submit`.
///
/// # Ordering contract (the one way to get this wrong)
///
/// Exporting a `SYNC_FD` payload requires a **pending or completed signal
/// operation**, and the export *resets* the semaphore. So the only correct
/// order is
///
/// ```text
///   arm(queue)  →  queue.submit(..)  →  export()
/// ```
///
/// one for one, every frame. Get it wrong and there is no gentle failure:
///
/// * Export with **nothing queued to signal** and the spec's answer is `-1`
///   ([`Error::NoPendingSignal`]) — but the answer measured on Mesa `anv` 25.3
///   is that `vkGetSemaphoreFdKHR` **blocks forever** on a syncobj that will
///   never be signalled. That is a hung thread inside the ICD, with only a
///   validation-layer message to explain it.
/// * Export **after** the work has already retired and the fd is born
///   signalled: it fences nothing, and every check short of "was it unsignalled
///   at submit time?" still passes.
///
/// So prefer [`Self::export_for_submit`], which makes the order structurally
/// impossible to get wrong.
///
/// One fence object is enough for the life of a process: export resets it, so
/// it is re-armable every frame.
pub struct CompletionFence {
    raw_device: ash::Device,
    ext: ash::khr::external_semaphore_fd::Device,
    semaphore: vk::Semaphore,
}

impl fmt::Debug for CompletionFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompletionFence")
            .field("semaphore", &self.semaphore)
            .finish_non_exhaustive()
    }
}

impl CompletionFence {
    /// Mint an export-capable binary `VkSemaphore`.
    ///
    /// Fails loudly if `VK_KHR_external_semaphore_fd` is not enabled
    /// ([`Error::NoExternalSemaphoreFd`]) or if the driver will not *export*
    /// `SYNC_FD` handles.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let (raw_device, instance, pdev) = semaphore_ctx(device)?;
        if !external_semaphore_features(&instance, pdev)
            .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
        {
            return Err(Error::SyncFdExportUnsupported);
        }
        let ext = ash::khr::external_semaphore_fd::Device::new(&instance, &raw_device);
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let create = vk::SemaphoreCreateInfo::default().push_next(&mut export);
        // SAFETY: a well-formed create-info chain on a live device.
        let semaphore = unsafe { raw_device.create_semaphore(&create, None) }
            .map_err(Error::vk("vkCreateSemaphore(exportable)"))?;
        Ok(Self {
            raw_device,
            ext,
            semaphore,
        })
    }

    /// The raw handle, for callers talking to Vulkan themselves.
    pub fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }

    /// Stage this fence to be signalled by the next `vkQueueSubmit` on `queue`.
    ///
    /// Read the type-level ordering contract: the *next* submit consumes it,
    /// whatever that submit turns out to be.
    pub fn arm(&self, queue: &wgpu::Queue) -> Result<()> {
        // SAFETY: staging only; nothing wgpu owns is destroyed.
        unsafe {
            let hal = queue.as_hal::<Vulkan>().ok_or(Error::NotVulkan)?;
            // `None` = binary semaphore. A timeline value here would be a
            // different (and for SYNC_FD export, illegal) mechanism.
            hal.add_signal_semaphore(self.semaphore, None);
        }
        Ok(())
    }

    /// Un-stage a signal that has not been consumed yet. Returns `true` if it
    /// was still pending.
    ///
    /// A frame that arms and then bails would otherwise leave a live signal for
    /// the next frame's submit to eat, and that frame's export would fence the
    /// wrong work.
    pub fn cancel_pending_signal(&self, queue: &wgpu::Queue) -> Result<bool> {
        // SAFETY: as above.
        unsafe {
            let hal = queue.as_hal::<Vulkan>().ok_or(Error::NotVulkan)?;
            Ok(hal.remove_signal_semaphore(self.semaphore))
        }
    }

    /// Export the pending (or completed) signal as a `sync_file`.
    ///
    /// Call **after** the submit that signals it. Exporting resets the
    /// semaphore, so this pairs one-for-one with [`Self::arm`].
    ///
    /// `-1` from the driver is reported as [`Error::NoPendingSignal`] rather
    /// than handed on as an already-signalled fence, because an
    /// already-signalled fence handed to a consumer is exactly the silent no-op
    /// this type exists to prevent. Do not rely on that error as a safety net:
    /// on `anv` the illegal call hangs instead of returning.
    pub fn export(&self) -> Result<SyncFile> {
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        // SAFETY: `self.semaphore` was created with SYNC_FD as an export type.
        let raw = unsafe { self.ext.get_semaphore_fd(&info) }
            .map_err(Error::vk("vkGetSemaphoreFdKHR"))?;
        if raw < 0 {
            return Err(Error::NoPendingSignal);
        }
        // SAFETY: the driver just handed us ownership of this fd.
        Ok(SyncFile::from_fd(unsafe { OwnedFd::from_raw_fd(raw) }))
    }

    /// The whole contract in one call: arm, run `submit`, export.
    ///
    /// This cannot be called in the wrong order. `submit` must issue exactly
    /// one `vkQueueSubmit` on `queue` (a `wgpu::Queue::submit`); its return
    /// value is passed back untouched so callers keep their `SubmissionIndex`.
    /// If the export fails the staged signal is withdrawn, so the next frame
    /// starts clean.
    pub fn export_for_submit<R>(
        &self,
        queue: &wgpu::Queue,
        submit: impl FnOnce() -> R,
    ) -> Result<(SyncFile, R)> {
        self.arm(queue)?;
        let out = submit();
        match self.export() {
            Ok(fd) => Ok((fd, out)),
            Err(e) => {
                let _ = self.cancel_pending_signal(queue);
                Err(e)
            }
        }
    }
}

impl Drop for CompletionFence {
    fn drop(&mut self) {
        // SAFETY: created here. Callers are required to have drained the queue.
        unsafe { self.raw_device.destroy_semaphore(self.semaphore, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ioctl number is the one thing here that cannot be checked by running
    /// the code — a wrong number returns `ENOTTY` on some fds and does
    /// something *else* on others. So it is recomputed from the `_IOWR`
    /// formula rather than trusted as a literal.
    #[test]
    fn export_sync_file_ioctl_matches_the_iowr_formula() {
        const IOC_WRITE: libc::c_ulong = 1;
        const IOC_READ: libc::c_ulong = 2;
        let size = core::mem::size_of::<ExportSyncFileArg>() as libc::c_ulong;
        assert_eq!(
            size, 8,
            "struct dma_buf_export_sync_file is two 32-bit words"
        );
        let computed =
            ((IOC_READ | IOC_WRITE) << 30) | (size << 16) | ((b'b' as libc::c_ulong) << 8) | 2;
        assert_eq!(computed, DMA_BUF_IOCTL_EXPORT_SYNC_FILE);
    }

    /// Same argument for the second ioctl number. A wrong number here would not
    /// report a wrong fence, it would report *no* fences — which [`fence_info`]
    /// returns as an empty vector, i.e. it would fail silently in exactly the
    /// direction that makes a stub look like an absence.
    #[test]
    fn sync_file_info_ioctl_matches_the_iowr_formula() {
        const IOC_WRITE: libc::c_ulong = 1;
        const IOC_READ: libc::c_ulong = 2;
        let size = core::mem::size_of::<SyncFileInfo>() as libc::c_ulong;
        assert_eq!(size, 56, "struct sync_file_info is 56 bytes");
        assert_eq!(
            core::mem::size_of::<SyncFenceInfoRaw>(),
            80,
            "struct sync_fence_info is 80 bytes"
        );
        let computed =
            ((IOC_READ | IOC_WRITE) << 30) | (size << 16) | ((b'>' as libc::c_ulong) << 8) | 4;
        assert_eq!(computed, SYNC_IOC_FILE_INFO);
    }

    /// The import ioctl, recomputed from `_IOW` the same way.
    #[test]
    fn import_sync_file_ioctl_matches_the_iow_formula() {
        const IOC_WRITE: libc::c_ulong = 1;
        let size = core::mem::size_of::<ImportSyncFileArg>() as libc::c_ulong;
        assert_eq!(
            size, 8,
            "struct dma_buf_import_sync_file is two 32-bit words"
        );
        let computed = (IOC_WRITE << 30) | (size << 16) | ((b'b' as libc::c_ulong) << 8) | 3;
        assert_eq!(computed, DMA_BUF_IOCTL_IMPORT_SYNC_FILE);
    }

    /// Same refusal for the import: a fence attached to something that is
    /// not a dma-buf must be an error, or a caller would skip its CPU wait.
    #[test]
    fn a_character_device_takes_no_fence() {
        let f = std::fs::File::open("/dev/null").expect("open /dev/null");
        let err = import_sync_file(f.as_fd(), f.as_fd(), SyncDirection::Write)
            .expect_err("/dev/null has no dma-buf ioctls");
        match err {
            Error::Io(e) => assert_eq!(e.raw_os_error(), Some(libc::ENOTTY), "got {e}"),
            other => panic!("expected an io error, got {other}"),
        }
    }

    /// The direction flags are the half of this API that is easy to get exactly
    /// backwards, and getting it backwards is silent.
    #[test]
    fn read_direction_is_the_read_bit() {
        assert_eq!(SyncDirection::Read.flags(), 1);
        assert_eq!(SyncDirection::Write.flags(), 2);
    }

    /// `is_stub` must need both names: a driver is free to call a timeline
    /// whatever it likes, and the claim being made — "the reservation object
    /// was empty" — is a claim about `dma_fence_stub_ops`, which sets both.
    #[test]
    fn is_stub_needs_both_names() {
        let stub = FenceInfo {
            driver_name: "stub".into(),
            obj_name: "stub".into(),
            status: 1,
        };
        assert!(stub.is_stub());
        let real = FenceInfo {
            driver_name: "i915".into(),
            obj_name: "stub".into(),
            status: 1,
        };
        assert!(!real.is_stub());
        assert_eq!(provenance_line(&[]), "<none reported>");
    }

    /// A non-dma-buf fd must be refused loudly, not silently treated as
    /// fenceless: if the ioctl quietly succeeded on anything, a caller could
    /// count waits against buffers that never had one.
    #[test]
    fn a_character_device_is_not_a_dmabuf() {
        let f = std::fs::File::open("/dev/null").expect("open /dev/null");
        let err = export_sync_file(f.as_fd(), SyncDirection::Read)
            .expect_err("/dev/null has no dma-buf ioctls");
        match err {
            Error::Io(e) => assert_eq!(e.raw_os_error(), Some(libc::ENOTTY), "got {e}"),
            other => panic!("expected an io error, got {other}"),
        }
    }
}
