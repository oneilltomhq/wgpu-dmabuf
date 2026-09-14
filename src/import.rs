//! dma-buf → `wgpu::Texture`.

use crate::dmabuf::Dmabuf;
use crate::error::{Error, Result};
use crate::format;

use wgpu::hal::api::Vulkan;

/// The initial tracker state this crate gives an imported texture.
///
/// **`RESOURCE | COPY_SRC`, and the choice is load-bearing.** Two separate
/// facts pin it:
///
/// * `wgpu::Device::create_texture_from_hal` asserts on a multi-bit initial
///   state that contains a *write* (`RESOURCE | COPY_SRC | COPY_DST` trips a
///   `strict_assert` in wgpu-core's `track/texture.rs`). The assert is
///   debug-build-only — a release build sails past it into undefined tracker
///   state, which is the worse half of the finding.
/// * `TextureUses::RESOURCE` alone maps to `SHADER_READ_ONLY_OPTIMAL`, while a
///   *combination* of bits maps to `VK_IMAGE_LAYOUT_GENERAL`
///   (wgpu-hal `vulkan::conv::derive_image_layout`). `GENERAL` is the natural
///   resting layout for an image shared with another API, so the second bit is
///   how you tell wgpu the truth about a buffer it did not allocate.
pub const IMPORT_STATE: wgpu::TextureUses =
    wgpu::TextureUses::RESOURCE.union(wgpu::TextureUses::COPY_SRC);

/// Import a dma-buf as a `wgpu::Texture`.
///
/// `usage` is the wgpu-level usage the returned texture will carry. The
/// texture's *initial tracker state* is [`IMPORT_STATE`] regardless; use
/// [`import_with_state`] if you need a different one (a texture you intend to
/// render into wants `UNINITIALIZED`, not a read-only state).
///
/// # Ownership
///
/// The plane's fd is **consumed**: on success Vulkan owns it, and on failure it
/// is closed. If the caller needs to keep an fd (because smithay, gbm or a
/// Wayland client also holds the buffer), pass [`Dmabuf::try_clone`]'s output —
/// duplicating a dma-buf fd is cheap and the duplicates name the same memory.
///
/// # Requirements
///
/// * The device must be Vulkan and must have been created with
///   [`wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF`] — see
///   [`crate::device`] for a helper that gets this right.
/// * Exactly one plane. Multi-planar YUV is refused, loudly
///   ([`Error::MultiPlaneUnsupported`]).
/// * [`Dmabuf::modifier`] must be the buffer's real modifier.
///
/// # Synchronisation
///
/// Importing establishes *sharing*, not *ordering*. Nothing about the returned
/// texture waits for the producer's GPU work. On Mesa `anv` an implicit-sync
/// conjunction (i915 KMD, dedicated allocation, `COLOR_ATTACHMENT` producer,
/// imported memory) happens to order it for you, which is why a naive test
/// passes; that conjunction breaks under the `xe` KMD, a drirc flip, or a
/// compute producer, and Mesa documents implicit sync as deprecated. **Treat a
/// passing unfenced test as proving nothing.** Use [`crate::sync`] to make the
/// ordering explicit.
pub fn import(
    device: &wgpu::Device,
    buf: &Dmabuf,
    usage: wgpu::TextureUsages,
) -> Result<wgpu::Texture> {
    import_with_state(device, buf, usage, IMPORT_STATE)
}

/// [`import`] with an explicit initial tracker state.
///
/// Read [`IMPORT_STATE`] before passing anything else. The states that are
/// actually safe here are a read-only combination (the default) or
/// `TextureUses::UNINITIALIZED` for a texture the caller is about to render
/// into and does not want wgpu to assume anything about.
pub fn import_with_state(
    device: &wgpu::Device,
    buf: &Dmabuf,
    usage: wgpu::TextureUsages,
    initial_state: wgpu::TextureUses,
) -> Result<wgpu::Texture> {
    let plane = buf.sole_plane()?;
    let wgpu_format = format::wgpu_format(buf.fourcc).ok_or(Error::UnmappableFourcc(buf.fourcc))?;

    if !device
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return Err(Error::MissingDmabufFeature);
    }

    let size = buf.extent();
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some("wgpu-dmabuf import"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu_format,
        usage: hal_uses(usage),
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    // Vulkan takes ownership of the fd on success and closes it on failure, so
    // hand it over by value and never touch it again either way.
    let fd = plane.fd.try_clone()?;
    let stride = u64::from(plane.stride);
    let offset = u64::from(plane.offset);

    let hal_texture = {
        // SAFETY: we only read the hal device out of `device` and hand it a
        // descriptor that matches the dma-buf we were given. Nothing wgpu owns
        // is destroyed here.
        let hal_device = unsafe { device.as_hal::<Vulkan>() }.ok_or(Error::NotVulkan)?;
        // SAFETY: `fd` is a dma-buf fd owned by us and matching `hal_desc`;
        // `modifier`/`stride`/`offset` are the layout the producer reported.
        unsafe { hal_device.texture_from_dmabuf_fd(fd, &hal_desc, buf.modifier, stride, offset) }
            .map_err(|e| Error::Hal(format!("texture_from_dmabuf_fd: {e:?}")))?
    };

    let desc = wgpu::TextureDescriptor {
        label: Some("wgpu-dmabuf import"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu_format,
        usage,
        view_formats: &[],
    };

    // SAFETY: `hal_texture` was just created by this device's hal, matches
    // `desc`, and is in the state we declare (see `IMPORT_STATE`).
    Ok(unsafe { device.create_texture_from_hal::<Vulkan>(hal_texture, &desc, initial_state) })
}

/// The hal-level usage bits implied by a set of wgpu-level usage bits.
///
/// wgpu-core derives this internally for textures it allocates, but an imported
/// texture is created through the hal directly and has to be told. The
/// storage-texture row picks `STORAGE_READ_WRITE` because wgpu's
/// `STORAGE_BINDING` does not distinguish, and a narrower guess would be a
/// usage the image was not created with.
pub(crate) fn hal_uses(usage: wgpu::TextureUsages) -> wgpu::TextureUses {
    let mut uses = wgpu::TextureUses::empty();
    if usage.contains(wgpu::TextureUsages::COPY_SRC) {
        uses |= wgpu::TextureUses::COPY_SRC;
    }
    if usage.contains(wgpu::TextureUsages::COPY_DST) {
        uses |= wgpu::TextureUses::COPY_DST;
    }
    if usage.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
        uses |= wgpu::TextureUses::RESOURCE;
    }
    if usage.contains(wgpu::TextureUsages::STORAGE_BINDING) {
        uses |= wgpu::TextureUses::STORAGE_READ_WRITE;
    }
    if usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        uses |= wgpu::TextureUses::COLOR_TARGET;
    }
    uses
}

/// The `VkImageUsageFlags` implied by a set of wgpu-level usage bits.
///
/// Needed by the modifier queries and by the export path, both of which speak
/// Vulkan rather than wgpu.
pub(crate) fn vk_usage(usage: wgpu::TextureUsages) -> ash::vk::ImageUsageFlags {
    use ash::vk::ImageUsageFlags as U;
    let mut out = U::empty();
    if usage.contains(wgpu::TextureUsages::COPY_SRC) {
        out |= U::TRANSFER_SRC;
    }
    if usage.contains(wgpu::TextureUsages::COPY_DST) {
        out |= U::TRANSFER_DST;
    }
    if usage.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
        out |= U::SAMPLED;
    }
    if usage.contains(wgpu::TextureUsages::STORAGE_BINDING) {
        out |= U::STORAGE;
    }
    if usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        out |= U::COLOR_ATTACHMENT;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The initial state must be read-only *and* multi-bit: read-only because
    /// wgpu-core asserts otherwise, multi-bit because that is what selects
    /// `VK_IMAGE_LAYOUT_GENERAL`.
    #[test]
    fn import_state_is_read_only_and_multi_bit() {
        assert!(!IMPORT_STATE.intersects(wgpu::TextureUses::EXCLUSIVE));
        assert!(IMPORT_STATE.bits().count_ones() > 1);
    }

    #[test]
    fn usage_bits_map_across() {
        let u = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC;
        assert_eq!(hal_uses(u), IMPORT_STATE);
        assert_eq!(
            vk_usage(u),
            ash::vk::ImageUsageFlags::SAMPLED | ash::vk::ImageUsageFlags::TRANSFER_SRC
        );
    }
}
