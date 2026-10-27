# wgpu-dmabuf

Linux **dma-buf interop for [wgpu](https://wgpu.rs)**, on the Vulkan backend.

| | |
|---|---|
| `import` | a dma-buf fd → a `wgpu::Texture` you can sample |
| `export` | a `wgpu::Texture` you can render into → a dma-buf fd |
| `modifiers` / `shareable` | which DRM format modifiers this adapter will actually share |
| `sync` | `sync_file` fences across devices and processes |
| `device::open_device` | a `VkDevice` that can do all four |
| `device::render_node` | the adapter's DRM render node as a `dev_t`, for `zwp_linux_dmabuf_feedback_v1.main_device` |

It is a thin layer of [`ash`](https://crates.io/crates/ash) against the
`VkDevice` wgpu already owns — no fork of wgpu, no patched branch, no second
device.

```rust
use wgpu_dmabuf::{Dmabuf, format::{Fourcc, LINEAR}};

// A buffer a Wayland client, GBM, or another process handed you.
let buf = Dmabuf::single_plane(w, h, Fourcc::Argb8888, LINEAR, fd, 0, stride);
let texture = wgpu_dmabuf::import(&device, &buf, wgpu::TextureUsages::TEXTURE_BINDING)?;
```

## Does this need a patched wgpu? No.

This crate builds against **crates.io `wgpu = "30.0.1"`** and nothing else. The
question was worth asking, because the prior art this crate was extracted from
ran against a vendored wgpu branch. The answer, checked against the wgpu-hal
30.0.1 source (gfx-rs/wgpu tag [`v30.0.1`](https://github.com/gfx-rs/wgpu/tree/v30.0.1/wgpu-hal)):

| what | in crates.io wgpu-hal 30.0.1? |
|---|---|
| `vulkan::Device::texture_from_dmabuf_fd` | **yes** — [`src/vulkan/device.rs:525`](https://github.com/gfx-rs/wgpu/blob/v30.0.1/wgpu-hal/src/vulkan/device.rs#L525), single-plane, gated on `Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF` |
| `vulkan::Device::texture_from_raw` + `TextureMemory::External` | **yes** |
| `vulkan::Queue::add_wait_semaphore` / `add_signal_semaphore` / `remove_*` | **yes** — [`src/vulkan/mod.rs:1524-1574`](https://github.com/gfx-rs/wgpu/blob/v30.0.1/wgpu-hal/src/vulkan/mod.rs#L1524-L1574) |
| `Adapter::device_from_raw`, `required_device_extensions`, `physical_device_features` | **yes** |
| `VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier` enabled by `request_device` | **yes**, when the driver has them |
| a dma-buf **export** helper | **no** — done here with raw `ash` |
| DRM modifier **enumeration** | **no** — done here with raw `ash` |
| `VK_KHR_external_semaphore_fd` **enabled on the device** | **no** — the string appears nowhere in the crate |

So import needs no raw-ash route (the upstream helper landed), while export,
modifier enumeration and the fence extension do. That last row is the reason
`device::open_device` exists: the `Queue::add_wait_semaphore` seam is upstream,
but the extension that makes `vkImportSemaphoreFdKHR` callable is not enabled,
so `open_device` calls `vkCreateDevice` itself with the extension added and
hands the result back through `Adapter::device_from_raw`. A regression test
(`stock_request_device_cannot_fence`) fails the day upstream changes this.

## Sharing is not ordering

Importing a dma-buf tells the driver where the bytes are. It says nothing about
*when* they are valid.

On Mesa `anv` a narrow implicit-sync conjunction — i915 KMD, dedicated
allocation, `COLOR_ATTACHMENT` producer, imported memory — supplies the
ordering for free, which is exactly why a naive unfenced test passes and
**proves nothing**. That conjunction breaks under the `xe` KMD, under a drirc
flip, and under a compute producer, and Mesa documents implicit sync as
deprecated. Anywhere a dma-buf crosses a device boundary, fence it explicitly
with `wgpu_dmabuf::sync`.

The `sync` module carries two further traps in its own docs:

* `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` **never fails for want of a fence** — an
  empty reservation object yields an already-signalled stub and a perfectly
  valid fd. `ExportedFence::already_signalled` is the bit that separates "the
  wait had teeth" from "the wait was free", and it is a struct field rather
  than a log line so a harness can fail on it.
* `CompletionFence` must be used **arm → submit → export**, one for one.
  Exporting with nothing queued to signal is specified to return `-1`, but on
  Mesa `anv` 25.3 it *blocks forever* inside the ICD. `export_for_submit` makes
  the order structurally impossible to get wrong.

## Driver-conditional behaviour, recorded rather than smoothed over

Each of these is in the doc comment of the item it affects, not just here.

* **`wgpu-core` asserts on the initial tracker state of an imported texture.**
  A multi-bit initial state containing a *write* trips a `strict_assert` in
  `track/texture.rs`; a release build sails past it into undefined tracker
  state. `import` uses `RESOURCE | COPY_SRC` — read-only, and multi-bit so that
  wgpu-hal's `derive_image_layout` selects `VK_IMAGE_LAYOUT_GENERAL`, the
  resting layout for a shared image.
* **`gbm_bo_create(USE_WRITE | USE_LINEAR)` returns NULL (`EINVAL`)** on Mesa
  25.3 `iris` via a render node. The test helper uses
  `gbm_bo_create_with_modifiers2` with an explicit `[LINEAR]` list and
  `USE_RENDERING`, filled through `gbm_bo_map`.
* **A signalled `dma_fence` has no legible name on kernel 7.1.**
  `dma_fence_driver_name` / `dma_fence_timeline_name` return
  `"detached-driver"` / `"signaled-timeline"` for *any* signalled fence without
  consulting the ops table, so `FenceInfo::is_stub` cannot fire for the stub it
  was written to detect. Only the *positive* reading — the kernel named
  something, therefore this is not the stub — is available, and only on an
  unsignalled export.
* **`X`-alpha fourccs map to the same Vulkan format as their `A` twins.** There
  is no ignore-the-alpha `UNORM` format, so `XRGB8888` samples the alpha byte
  rather than ignoring it.
* **Half-float formats are unproven.** `ABGR16161616F → Rgba16Float` is the
  obvious mapping and no more than that; nothing in the test suite exercises it.

## Not implemented

* **Multi-planar formats** (NV12 and friends). `texture_from_dmabuf_fd` is
  single-plane, and the multi-plane route needs a `VkSamplerYcbcrConversion`
  that has no representation in wgpu's texture model. A multi-plane
  description is refused with `Error::MultiPlaneUnsupported` rather than
  silently imported as its luma plane.
* **Timeline semaphores.** `add_wait_semaphore` accepts a timeline value and
  `VK_KHR_external_semaphore_fd` can carry `OPAQUE_FD` timeline handles, but
  `SYNC_FD` is binary-only, nothing here has ever run a timeline export, and an
  unproven fence is worse than an absent one.
* **KMS scanout.** A `sync_file` from `CompletionFence` is exactly the fd an
  atomic commit wants for `IN_FENCE_FD`, but this crate does not talk to KMS.
* **Discrete GPUs and cross-vendor handoffs.** Untested.

## Testing

```
cargo test --features gbm
```

The integration tests mint their dma-bufs with GBM, so without `--features gbm`
only the unit and doc tests are built.

Tests that need a GPU **skip** (they do not fail) when `/dev/dri/renderD*` is
absent or no Vulkan adapter can be opened — "there is no GPU here" and "the GPU
got it wrong" must not look alike.

Measured on an Intel Iris Xe Graphics (RPL-U), Mesa 25.3.6, kernel 7.1.9,
`/dev/dri/renderD128` — **23 passed, 0 failed, 0 skipped**
(14 unit + 8 integration + 1 doc):

* `import_a_gbm_dmabuf_and_sample_it_byte_exact` — a 256×256 `ARGB8888` LINEAR
  buffer allocated and filled by GBM on the CPU, imported, sampled with
  `textureLoad` in WGSL, read back: **byte-exact over all 65 536 pixels**.
* `export_a_texture_render_into_it_and_reimport_byte_exact` — an exportable
  image (the driver chose modifier `0x0100000000000002`, a real tiled layout),
  a WGSL pass renders a pattern into it, the exported fd is imported *back*
  through the public import path and sampled: **byte-exact**. A wrong modifier,
  stride or offset fails here and would not fail a plain
  `copy_texture_to_buffer`.
* `a_completion_fence_attaches_to_a_dmabuf_as_its_write_fence` — a completion
  fence imported into a GBM dma-buf with `IMPORT_SYNC_FILE`; `EXPORT_SYNC_FILE`
  for a reader then hands back that fence (provenance
  `driver="0000:00:02.0"`, unsignalled at export), and it signals.
* `a_completion_fence_exports_a_sync_file_that_signals` — arm → submit →
  export, and the fd signals within a bounded poll. Provenance read back from
  the fd at export time: `driver="0000:00:02.0" obj="…[pid]" status=0` — an
  unsignalled i915 fence, i.e. the positive reading, not a stub.
* `an_acquire_semaphore_accepts_an_exported_fence` — the fence imports as a
  temporary `SYNC_FD` payload and is consumed by the next submit. This proves
  the plumbing, not the ordering; no timing test can tell a real wait from a
  wait on already-retired work.
* `modifiers_are_enumerable_and_shareable_is_a_subset` — 5 modifiers for
  `ARGB8888`, 3 of them single-plane and dma-buf shareable both ways.
* `stock_request_device_cannot_fence` — the finding above, as a regression test.
* `the_render_node_is_this_adapters` — `device::render_node` names
  `226:128`, checked against the kernel rather than against itself:
  `/dev/dri/renderD128` carries that `dev_t`, and its sysfs PCI vendor is the
  adapter's.
* Unit tests recompute both `_IOWR` ioctl numbers from the formula, pin the
  fourcc channel order, and check that a non-dma-buf fd is refused with
  `ENOTTY` rather than silently treated as fenceless.

## Versions

| | |
|---|---|
| `wgpu` | `30.0.1` (features `std`, `vulkan`, `wgsl`) |
| `wgpu-hal` | `=30.0.1`, feature `vulkan` — pinned exactly so the hal types crossing the API are *the same* crate instance wgpu was built against |
| `ash` | `0.38` |
| `gbm` | `0.18` (optional, `gbm` feature, test helper only) |
| `drm-fourcc` | `2` |
| MSRV | 1.87 |

## License

MIT
