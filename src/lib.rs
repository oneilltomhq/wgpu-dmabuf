//! Linux dma-buf interop for [`wgpu`], on the Vulkan backend.
//!
//! This is the vocabulary layer: the way a dma-buf is *described* (a fourcc, a
//! DRM format modifier, and one plane's fd, offset and stride), the mapping
//! between DRM fourccs and `wgpu::TextureFormat`, and the error type everything
//! else returns.

#![cfg(target_os = "linux")]
#![deny(missing_debug_implementations)]
#![warn(missing_docs)]

pub mod dmabuf;
pub mod error;
pub mod format;

pub use dmabuf::{Dmabuf, DmabufPlane};
pub use error::{Error, Result};
pub use format::Fourcc;

/// The `ash` this crate is built against, re-exported so a caller names *the*
/// `ash`, not a semver-compatible second copy.
pub use ash;
