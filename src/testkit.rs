//! A GBM-backed allocator, for tests and examples. Requires the `gbm` feature.
//!
//! Nothing in the import path can be graded without a *real* dma-buf whose
//! contents are known independently of the GPU. This module allocates one on a
//! DRM render node, fills it on the CPU, and hands back a [`Dmabuf`].
//!
//! # The allocation call is not the obvious one
//!
//! `gbm_bo_create(width, height, format, GBM_BO_USE_WRITE | GBM_BO_USE_LINEAR)`
//! — the call every tutorial starts with — **returns NULL (`EINVAL`) on Mesa
//! 25.3 `iris` via a render node**. The working call is
//! `gbm_bo_create_with_modifiers2` with an explicit `[LINEAR]` modifier list and
//! `GBM_BO_USE_RENDERING`, filled through `gbm_bo_map` rather than
//! `gbm_bo_write`. That is what [`GbmAllocator::allocate_linear`] does.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use crate::dmabuf::Dmabuf;
use crate::format::Fourcc;

/// The first DRM render node on this machine, if there is one.
///
/// Tests that need a GPU should skip — not fail — when this returns `None`:
/// "there is no GPU here" and "the GPU got it wrong" are different answers and
/// must not look alike in CI output.
pub fn render_node() -> Option<PathBuf> {
    let mut nodes: Vec<PathBuf> = std::fs::read_dir("/dev/dri")
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("renderD"))
        })
        .collect();
    nodes.sort();
    nodes.into_iter().next()
}

/// A GBM device on a DRM render node.
#[derive(Debug)]
pub struct GbmAllocator {
    device: gbm::Device<File>,
}

/// A GBM buffer object and the [`Dmabuf`] describing it.
///
/// Holds the `gbm::BufferObject` for the lifetime of the description. The
/// exported fd would keep the memory alive on its own, but keeping the bo means
/// [`GbmBuffer::read_back`] can still map it.
#[derive(Debug)]
pub struct GbmBuffer {
    bo: gbm::BufferObject<()>,
    /// The dma-buf, ready to hand to [`crate::import()`].
    pub dmabuf: Dmabuf,
}

impl GbmAllocator {
    /// Open a specific render node.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let device = gbm::Device::new(file)?;
        Ok(Self { device })
    }

    /// Open [`render_node`], or `None` if this machine has no render node.
    pub fn open_default() -> Option<Self> {
        Self::open(&render_node()?).ok()
    }

    /// Allocate a `LINEAR` single-plane buffer and fill it on the CPU.
    ///
    /// `fill` is called once per pixel and returns the four bytes **in memory
    /// order** — for [`Fourcc::Argb8888`] that is `[B, G, R, A]`. Writing the
    /// channels in fourcc-name order is the classic way to produce a
    /// convincingly wrong picture.
    ///
    /// `LINEAR` is not a preference here, it is a requirement: a tiled buffer's
    /// bytes are not in raster order, so filling one row-by-row yields a
    /// swizzled image that no assertion can interpret.
    pub fn allocate_linear(
        &self,
        width: u32,
        height: u32,
        fourcc: Fourcc,
        mut fill: impl FnMut(u32, u32) -> [u8; 4],
    ) -> io::Result<GbmBuffer> {
        let format = gbm::Format::try_from(fourcc as u32)
            .map_err(|_| io::Error::other(format!("gbm does not know {fourcc}")))?;
        let mut bo = self
            .device
            .create_buffer_object_with_modifiers2::<()>(
                width,
                height,
                format,
                [gbm::Modifier::Linear].into_iter(),
                gbm::BufferObjectFlags::RENDERING,
            )
            .map_err(|e| io::Error::other(format!("gbm_bo_create_with_modifiers2: {e}")))?;

        bo.map_mut(0, 0, width, height, |mm| {
            let stride = mm.stride();
            let buf = mm.buffer_mut();
            for y in 0..height {
                for x in 0..width {
                    let o = (y * stride + x * 4) as usize;
                    buf[o..o + 4].copy_from_slice(&fill(x, y));
                }
            }
        })
        .map_err(|e| io::Error::other(format!("gbm_bo_map: {e}")))?;

        let stride = bo.stride();
        let offset = bo.offset(0);
        let modifier: u64 = bo.modifier().into();
        let fd = bo.fd().map_err(io::Error::other)?;

        Ok(GbmBuffer {
            dmabuf: Dmabuf::single_plane(width, height, fourcc, modifier, fd, offset, stride),
            bo,
        })
    }
}

impl GbmBuffer {
    /// Read the buffer back through `gbm_bo_map`, row by row, into a tightly
    /// packed `width * height * 4` byte vector in memory order.
    ///
    /// The CPU-side counterpart to the GPU-side grader: this is how you check
    /// what a render pass actually wrote into an *exported* dma-buf without
    /// asking the same GPU that wrote it.
    pub fn read_back(&mut self) -> io::Result<Vec<u8>> {
        let (w, h) = (self.dmabuf.width, self.dmabuf.height);
        let mut out = vec![0u8; (w * h * 4) as usize];
        self.bo
            .map(0, 0, w, h, |mm| {
                let stride = mm.stride();
                let buf = mm.buffer();
                for y in 0..h {
                    let src = (y * stride) as usize;
                    let dst = (y * w * 4) as usize;
                    out[dst..dst + (w * 4) as usize]
                        .copy_from_slice(&buf[src..src + (w * 4) as usize]);
                }
            })
            .map_err(|e| io::Error::other(format!("gbm_bo_map: {e}")))?;
        Ok(out)
    }
}
