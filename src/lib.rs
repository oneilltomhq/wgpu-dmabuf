//! Linux dma-buf interop for [`wgpu`], on the Vulkan backend.
//!
//! | | |
//! |---|---|
//! | [`import()`] | a dma-buf fd → a `wgpu::Texture` you can sample |
//! | [`export()`] | a `wgpu::Texture` you can render into → a dma-buf fd |
//! | [`modifiers()`] | which DRM format modifiers this adapter will actually share |
//!
//! # Vulkan only, and deliberately
//!
//! Every entry point starts with `as_hal::<Vulkan>()` and returns
//! [`Error::NotVulkan`] if that fails. dma-buf is a Linux/DRM concept with a
//! Vulkan expression (`VK_EXT_external_memory_dma_buf` +
//! `VK_EXT_image_drm_format_modifier`); there is nothing to abstract over.
//!
//! # Sharing is not ordering
//!
//! Importing a dma-buf tells the driver where the bytes are. It says nothing
//! about *when* they are valid. Fencing lands in a later commit; until then,
//! treat a passing unfenced test as proving nothing.

#![cfg(target_os = "linux")]
#![deny(missing_debug_implementations)]
#![warn(missing_docs)]

pub mod dmabuf;
pub mod error;
pub mod export;
pub mod format;
pub mod import;
pub mod modifiers;

pub use dmabuf::{Dmabuf, DmabufPlane};
pub use error::{Error, Result};
pub use export::{export, ExportDescriptor};
pub use format::Fourcc;
pub use import::{import, import_with_state, IMPORT_STATE};
pub use modifiers::{modifiers, shareable, Modifier};

/// The `ash` this crate is built against, re-exported so a caller names *the*
/// `ash`, not a semver-compatible second copy.
pub use ash;
