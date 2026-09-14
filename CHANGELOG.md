# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First cut. Everything below is new.

### Added

- `import()` / `import_with_state()` — a single-plane dma-buf fd becomes a
  `wgpu::Texture`, via wgpu-hal 30.0.1's own `texture_from_dmabuf_fd`. The
  initial tracker state is `RESOURCE | COPY_SRC`, which is both read-only (a
  multi-bit *write* state trips a `strict_assert` in wgpu-core) and multi-bit
  (so that wgpu-hal derives `VK_IMAGE_LAYOUT_GENERAL`).
- `export()` — an exportable `VkImage` with a driver-chosen DRM modifier,
  wrapped as a `wgpu::Texture` you can render into, plus the `Dmabuf`
  description of it. No upstream helper exists for this, so it is raw `ash`
  plus `texture_from_raw(.., TextureMemory::External)` with a drop callback
  that destroys the image and frees the memory.
- `modifiers()` and `shareable()` — DRM format modifier enumeration, and the
  narrower list the adapter will actually import *and* export as a dma-buf.
  `best_for_gpu()` picks from the latter.
- `sync` — `sync_file` fencing. `CompletionFence` (arm → submit → export),
  `AcquireSemaphore` (import a `sync_file`, wait on the next submit),
  `export_sync_file()` straight off a dma-buf fd, `SyncFile` with a bounded
  `poll(2)` wait, and `FenceInfo` provenance via `SYNC_IOC_FILE_INFO`.
  `ExportedFence::already_signalled` reports the vacuous-wait case, because
  `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` never fails for want of a fence.
- `device::open_device()` — a `wgpu::Device` whose `VkDevice` has
  `VK_KHR_external_semaphore_fd` enabled. Stock `request_device` never enables
  it, so the fence path is uncallable without this.
- `Dmabuf` / `DmabufPlane` description types, a `format` mapping table between
  DRM fourccs, `wgpu::TextureFormat` and `VkFormat`, and a non-exhaustive
  `Error`.
- `testkit` (feature `gbm`) — a GBM allocator for tests, using
  `gbm_bo_create_with_modifiers2` with an explicit `[LINEAR]` list, because
  `gbm_bo_create(USE_WRITE | USE_LINEAR)` returns `EINVAL` on Mesa 25.3 `iris`.

### Known limitations

- Single-plane formats only; NV12 and friends are refused with
  `Error::MultiPlaneUnsupported`.
- No timeline semaphores. `SYNC_FD` is a binary handle type and nothing here
  has run a timeline export.
- Proven on one configuration only: Intel Iris Xe (RPL-U), Mesa 25.3.6, kernel
  7.1.9, i915. Discrete GPUs, the `xe` KMD and cross-vendor handoffs are
  untested.

[Unreleased]: https://github.com/oneilltomhq/wgpu-dmabuf
