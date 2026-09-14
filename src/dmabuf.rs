//! The description of a dma-buf: what you get off the wire, and what you need
//! to turn one into a `wgpu::Texture`.

use std::os::fd::{AsFd, BorrowedFd, OwnedFd};

use crate::error::{Error, Result};
use crate::format::Fourcc;

/// One plane of a dma-buf.
///
/// `fd` is owned. A dma-buf fd is a reference to the underlying buffer object,
/// so `dup(2)`ing one is cheap and correct: two fds naming the same memory is
/// the normal state of affairs, not a bug. What is *not* correct is handing the
/// same fd to Vulkan twice, because `vkAllocateMemory` takes ownership of a
/// `DMA_BUF` handle on success.
#[derive(Debug)]
pub struct DmabufPlane {
    /// The dma-buf file descriptor for this plane.
    pub fd: OwnedFd,
    /// Byte offset of the first pixel of this plane within the buffer object.
    pub offset: u32,
    /// Bytes between successive rows of this plane.
    pub stride: u32,
}

impl DmabufPlane {
    /// A plane sharing this plane's memory through a fresh fd.
    ///
    /// Use this whenever something else already owns the fd (smithay's
    /// `Dmabuf`, a `gbm::BufferObject`, a Wayland `wl_buffer`) — Vulkan will
    /// consume whatever it is given.
    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            fd: self.fd.try_clone()?,
            offset: self.offset,
            stride: self.stride,
        })
    }

    /// Borrow the fd, e.g. to hand to
    /// [`export_sync_file`](crate::sync::export_sync_file).
    pub fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

/// A dma-buf, described completely enough to import.
///
/// This is the same information a Wayland `zwp_linux_buffer_params_v1` carries
/// and the same information `gbm_bo_get_*` yields. It owns its fds.
///
/// # The modifier is not optional
///
/// `modifier` must be the real DRM format modifier the buffer was allocated
/// with, not `DRM_FORMAT_MOD_INVALID` and not a guess at `LINEAR`. Vulkan
/// imports a dma-buf through `VK_EXT_image_drm_format_modifier`, which has no
/// "I don't know" value: the modifier selects the tiling the driver will
/// *interpret the bytes as*. Get it wrong and the import succeeds and the
/// texture samples garbage.
#[derive(Debug)]
pub struct Dmabuf {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// DRM fourcc, e.g. [`Fourcc::Argb8888`].
    pub fourcc: Fourcc,
    /// DRM format modifier, e.g. [`crate::format::LINEAR`].
    pub modifier: u64,
    /// One entry per plane. This crate supports exactly one (see
    /// [`Error::MultiPlaneUnsupported`]).
    pub planes: Vec<DmabufPlane>,
}

impl Dmabuf {
    /// A single-plane dma-buf.
    pub fn single_plane(
        width: u32,
        height: u32,
        fourcc: Fourcc,
        modifier: u64,
        fd: OwnedFd,
        offset: u32,
        stride: u32,
    ) -> Self {
        Self {
            width,
            height,
            fourcc,
            modifier,
            planes: vec![DmabufPlane { fd, offset, stride }],
        }
    }

    /// The single plane, or the error explaining why there is not exactly one.
    pub fn sole_plane(&self) -> Result<&DmabufPlane> {
        match self.planes.len() {
            0 => Err(Error::NoPlanes),
            1 => Ok(&self.planes[0]),
            planes => Err(Error::MultiPlaneUnsupported { planes }),
        }
    }

    /// A description of the same memory with freshly `dup`ed fds.
    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            width: self.width,
            height: self.height,
            fourcc: self.fourcc,
            modifier: self.modifier,
            planes: self
                .planes
                .iter()
                .map(DmabufPlane::try_clone)
                .collect::<Result<Vec<_>>>()?,
        })
    }

    /// One line suitable for a log or a test failure message.
    pub fn describe(&self) -> String {
        let planes = self
            .planes
            .iter()
            .map(|p| format!("(off {} stride {})", p.offset, p.stride))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "{}x{} {} modifier 0x{:016x} planes {}",
            self.width, self.height, self.fourcc, self.modifier, planes
        )
    }

    /// The `wgpu::Extent3d` this buffer describes.
    pub fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        }
    }
}
