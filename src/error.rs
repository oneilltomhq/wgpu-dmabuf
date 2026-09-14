//! One error type for every way dma-buf interop can fail.
//!
//! There is deliberately no "carry on without it" variant. Every failure here
//! is a failure to establish a *guarantee* — that the memory is shared, that
//! the layout matches, that the wait has teeth — and degrading silently past
//! one of those produces plausible-looking wrong pixels rather than an error.

use std::fmt;

use crate::format::Fourcc;

/// Everything that can go wrong importing, exporting or fencing a dma-buf.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The `wgpu::Device`/`Adapter`/`Queue` is not backed by the Vulkan
    /// backend. This crate is Vulkan-only by construction: nothing in it has a
    /// meaning on Metal, D3D12 or GL.
    NotVulkan,

    /// The device was not created with
    /// [`wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF`].
    ///
    /// wgpu exposes that feature only when the driver has all three of
    /// `VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf` and
    /// `VK_EXT_image_drm_format_modifier`, and it must be *requested* at
    /// `request_device` time — an adapter that supports it does not imply a
    /// device that enabled it.
    MissingDmabufFeature,

    /// `VK_KHR_external_semaphore_fd` is not enabled on this `VkDevice`.
    ///
    /// Stock `wgpu::Adapter::request_device` never enables it (wgpu-hal 30.0.1
    /// mentions the extension nowhere). Open the device with
    /// [`crate::device::open_device`] instead, which adds it by hand.
    NoExternalSemaphoreFd,

    /// The driver refuses `SYNC_FD` as an importable external semaphore handle
    /// type, or refuses temporary import of it.
    SyncFdImportUnsupported,

    /// The driver refuses `SYNC_FD` as an *exportable* external semaphore
    /// handle type.
    SyncFdExportUnsupported,

    /// `vkGetSemaphoreFdKHR` handed back `-1`: the `SYNC_FD` convention for "no
    /// pending or completed signal operation to represent". Always a caller
    /// ordering bug — the signal must be queued on a submit *before* the
    /// export. See [`crate::sync::CompletionFence`].
    NoPendingSignal,

    /// The DRM fourcc has no `wgpu::TextureFormat` this crate is willing to
    /// claim equivalence with. See [`crate::format`].
    UnmappableFourcc(Fourcc),

    /// The `wgpu::TextureFormat` has no `VkFormat` this crate maps.
    UnmappableFormat(wgpu::TextureFormat),

    /// A multi-planar dma-buf was handed to a single-plane path.
    ///
    /// Multi-planar import (NV12 and friends) is *not* implemented here:
    /// `wgpu_hal::vulkan::Device::texture_from_dmabuf_fd` is single-plane only,
    /// and the multi-plane route needs a `VkSamplerYcbcrConversion` that has no
    /// representation in wgpu's texture model. Better an error than a texture
    /// that samples the luma plane and calls itself RGBA.
    MultiPlaneUnsupported {
        /// How many planes the description carried.
        planes: usize,
    },

    /// A dma-buf description with no planes at all.
    NoPlanes,

    /// No DRM format modifier in the requested list is both importable and
    /// exportable as a dma-buf for this format and usage.
    NoShareableModifier {
        /// The format that has no shareable modifier.
        fourcc: Fourcc,
    },

    /// No `VkMemoryType` satisfies both the image's requirements and the
    /// imported fd's. Usually a cross-device import the driver will not accept.
    NoCompatibleMemoryType,

    /// A Vulkan call failed.
    Vk {
        /// Which call, spelled as the Vulkan entry point.
        what: &'static str,
        /// The `VkResult`.
        result: ash::vk::Result,
    },

    /// A wgpu-hal call failed.
    Hal(String),

    /// `wgpu::Adapter::request_device` / `create_device_from_hal` failed.
    RequestDevice(String),

    /// An OS call failed — `ioctl`, `poll`, `dup`.
    Io(std::io::Error),
}

impl Error {
    pub(crate) fn vk(what: &'static str) -> impl FnOnce(ash::vk::Result) -> Self {
        move |result| Self::Vk { what, result }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotVulkan => write!(
                f,
                "not a wgpu-core Vulkan object; this crate is Vulkan-only"
            ),
            Self::MissingDmabufFeature => write!(
                f,
                "device was not created with Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF"
            ),
            Self::NoExternalSemaphoreFd => write!(
                f,
                "VK_KHR_external_semaphore_fd is not enabled on this VkDevice \
                 (stock request_device never enables it; use wgpu_dmabuf::device::open_device)"
            ),
            Self::SyncFdImportUnsupported => {
                write!(f, "driver does not support importing SYNC_FD semaphores")
            }
            Self::SyncFdExportUnsupported => {
                write!(f, "driver does not support exporting SYNC_FD semaphores")
            }
            Self::NoPendingSignal => write!(
                f,
                "vkGetSemaphoreFdKHR returned -1: no pending signal operation \
                 (arm the fence and submit before exporting)"
            ),
            Self::UnmappableFourcc(c) => write!(f, "no wgpu::TextureFormat for DRM format {c}"),
            Self::UnmappableFormat(fmt) => write!(f, "no VkFormat mapped for {fmt:?}"),
            Self::MultiPlaneUnsupported { planes } => write!(
                f,
                "multi-planar dma-buf ({planes} planes) is not supported; single-plane only"
            ),
            Self::NoPlanes => write!(f, "dma-buf description has no planes"),
            Self::NoShareableModifier { fourcc } => write!(
                f,
                "no DRM format modifier for {fourcc} is both dma-buf importable and exportable \
                 with the requested usage"
            ),
            Self::NoCompatibleMemoryType => {
                write!(
                    f,
                    "no VkMemoryType satisfies both the image and the imported fd"
                )
            }
            Self::Vk { what, result } => write!(f, "{what}: {result:?}"),
            Self::Hal(e) => write!(f, "wgpu-hal: {e}"),
            Self::RequestDevice(e) => write!(f, "request_device: {e}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// The crate's result alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;
