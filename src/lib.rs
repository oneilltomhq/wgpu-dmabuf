//! Linux dma-buf interop for [`wgpu`], on the Vulkan backend.
//!
//! Four things, each of which is a small amount of `ash` against the `VkDevice`
//! wgpu already owns:
//!
//! | | |
//! |---|---|
//! | [`import()`] | a dma-buf fd → a `wgpu::Texture` you can sample |
//! | [`export()`] | a `wgpu::Texture` you can render into → a dma-buf fd |
//! | [`modifiers()`] | which DRM format modifiers this adapter will actually share |
//! | [`sync`] | `sync_file` fences across devices and processes |
//!
//! ```no_run
//! use wgpu_dmabuf::{Dmabuf, format::Fourcc};
//!
//! # fn demo(device: &wgpu::Device, fd: std::os::fd::OwnedFd) -> Result<(), wgpu_dmabuf::Error> {
//! let buf = Dmabuf::single_plane(256, 256, Fourcc::Argb8888, wgpu_dmabuf::format::LINEAR, fd, 0, 1024);
//! let texture = wgpu_dmabuf::import(device, &buf, wgpu::TextureUsages::TEXTURE_BINDING)?;
//! # Ok(()) }
//! ```
//!
//! # Vulkan only, and deliberately
//!
//! Every entry point starts with `as_hal::<Vulkan>()` and returns
//! [`Error::NotVulkan`] if that fails. dma-buf is a Linux/DRM concept with a
//! Vulkan expression (`VK_EXT_external_memory_dma_buf` +
//! `VK_EXT_image_drm_format_modifier`); there is nothing to abstract over.
//!
//! # What this crate needs from wgpu, and what it does not
//!
//! Import needs [`wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF`], which stock
//! `request_device` will grant on a driver that supports it. Export needs the
//! same extensions but no wgpu API at all — it is `ash` plus
//! `texture_from_raw(.., TextureMemory::External)`.
//!
//! Fencing needs `VK_KHR_external_semaphore_fd`, which **stock
//! `request_device` never enables**. Use [`device::open_device`], which builds
//! the `VkDevice` by hand and hands it to wgpu. See [`sync`].
//!
//! # Sharing is not ordering
//!
//! Importing a dma-buf tells the driver where the bytes are. It says nothing
//! about *when* they are valid. On Mesa `anv` a narrow implicit-sync
//! conjunction (i915 KMD, dedicated allocation, `COLOR_ATTACHMENT` producer,
//! imported memory) supplies that ordering for free, which is why an unfenced
//! test passes and proves nothing — the conjunction breaks under the `xe` KMD,
//! under a drirc flip, and under a compute producer, and Mesa documents
//! implicit sync as deprecated. **Anywhere a dma-buf crosses a device boundary,
//! fence it explicitly.** [`sync`] is how.
//!
//! # What is not here
//!
//! * **Multi-planar formats** (NV12 and friends). Single plane only; a
//!   multi-plane description is refused with
//!   [`Error::MultiPlaneUnsupported`] rather than silently imported as its
//!   luma plane.
//! * **Timeline semaphores.** `SYNC_FD` is a binary-only handle type and
//!   nothing this crate was extracted from ran a timeline export. See [`sync`].
//! * **KMS scanout.** A `sync_file` from [`sync::CompletionFence`] is exactly
//!   the fd an atomic commit wants for `IN_FENCE_FD`, but this crate does not
//!   talk to KMS.

#![cfg(target_os = "linux")]
#![deny(missing_debug_implementations)]
#![warn(missing_docs)]
#![doc(html_root_url = "https://docs.rs/wgpu-dmabuf/0.1.0")]

pub mod device;
pub mod dmabuf;
pub mod error;
pub mod export;
pub mod format;
pub mod import;
pub mod modifiers;
pub mod sync;

#[cfg(feature = "gbm")]
pub mod testkit;

pub use dmabuf::{Dmabuf, DmabufPlane};
pub use error::{Error, Result};
pub use export::{export, ExportDescriptor};
pub use format::Fourcc;
pub use import::{import, import_with_state, IMPORT_STATE};
pub use modifiers::{modifiers, shareable, Modifier};

/// The `ash` this crate is built against, re-exported so a caller naming
/// `vk::PipelineStageFlags` for [`sync::AcquireSemaphore::wait_on_next_submit`]
/// names *the* `ash`, not a semver-compatible second copy.
pub use ash;
