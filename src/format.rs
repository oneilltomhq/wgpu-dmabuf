//! DRM fourcc ↔ `wgpu::TextureFormat` ↔ `VkFormat`.
//!
//! The mapping table is short on purpose. A fourcc and a `TextureFormat` agree
//! only if they agree on **byte order in memory**, and that is not something to
//! guess at: a wrong entry here produces a texture that samples with the
//! channels swapped, which is a picture, not an error.
//!
//! Only formats this crate is prepared to defend are listed. Each entry says
//! what evidence it rests on.

pub use drm_fourcc::{DrmFourcc as Fourcc, DrmModifier};

/// `DRM_FORMAT_MOD_LINEAR`. The one modifier every driver understands and the
/// only one a CPU writer (`gbm_bo_map`) can fill meaningfully.
pub const LINEAR: u64 = 0;

/// `DRM_FORMAT_MOD_INVALID` — "the allocator did not tell us". Never pass it
/// to an import: `VK_EXT_image_drm_format_modifier` has no notion of it, and a
/// driver handed `u64::MAX` as a modifier will refuse the image (at best).
pub const INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// The `wgpu::TextureFormat` a DRM fourcc imports as, or `None` if this crate
/// will not claim an equivalence.
///
/// DRM fourccs name channels in **little-endian word order** while wgpu names
/// them in **memory order**, which is why `ARGB8888` is `Bgra8Unorm` and not
/// `Rgba8Unorm`. Both 8888 rows are measured, not derived: `ABGR8888` LINEAR
/// imported as `Rgba8Unorm` samples byte-exact, and `ARGB8888` LINEAR as
/// `Bgra8Unorm` likewise.
///
/// The `X` variants map to the same `A` format: Vulkan has no
/// ignore-the-alpha-channel `UNORM` format, so the alpha byte is *sampled*
/// rather than ignored. If the producer left it at zero, a blend against that
/// texture will behave as fully transparent. Callers that care must force
/// alpha in the shader; this crate will not silently rewrite the format.
///
/// The half-float rows are **unproven** in this crate's test suite — they are
/// the obvious mapping and no more than that.
pub fn wgpu_format(fourcc: Fourcc) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as T;
    Some(match fourcc {
        // memory order B, G, R, A
        Fourcc::Argb8888 | Fourcc::Xrgb8888 => T::Bgra8Unorm,
        // memory order R, G, B, A
        Fourcc::Abgr8888 | Fourcc::Xbgr8888 => T::Rgba8Unorm,
        // unproven here
        Fourcc::Abgr16161616f | Fourcc::Xbgr16161616f => T::Rgba16Float,
        _ => return None,
    })
}

/// The `VkFormat` a `wgpu::TextureFormat` is created as by wgpu-hal.
///
/// Needed because the DRM-modifier queries (`vkGetPhysicalDeviceFormatProperties2`
/// with `VkDrmFormatModifierPropertiesListEXT`) are asked in Vulkan's
/// vocabulary, not wgpu's. Restricted to the formats [`wgpu_format`] can
/// produce plus their sRGB twins; anything else returns `None` rather than a
/// plausible guess.
pub fn vk_format(format: wgpu::TextureFormat) -> Option<ash::vk::Format> {
    use ash::vk::Format as V;
    use wgpu::TextureFormat as T;
    Some(match format {
        T::Rgba8Unorm => V::R8G8B8A8_UNORM,
        T::Rgba8UnormSrgb => V::R8G8B8A8_SRGB,
        T::Bgra8Unorm => V::B8G8R8A8_UNORM,
        T::Bgra8UnormSrgb => V::B8G8R8A8_SRGB,
        T::Rgba16Float => V::R16G16B16A16_SFLOAT,
        T::Rgb10a2Unorm => V::A2B10G10R10_UNORM_PACK32,
        _ => return None,
    })
}

/// The DRM fourcc a `wgpu::TextureFormat` exports as — the inverse of
/// [`wgpu_format`], restricted to the alpha-carrying spellings.
///
/// The `X` fourccs are deliberately not produced: exporting as `XRGB8888`
/// would be a promise that the alpha byte is meaningless, and a wgpu render
/// pass writes it.
pub fn fourcc_for(format: wgpu::TextureFormat) -> Option<Fourcc> {
    use wgpu::TextureFormat as T;
    Some(match format {
        T::Bgra8Unorm | T::Bgra8UnormSrgb => Fourcc::Argb8888,
        T::Rgba8Unorm | T::Rgba8UnormSrgb => Fourcc::Abgr8888,
        T::Rgba16Float => Fourcc::Abgr16161616f,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The channel order is the one thing here that cannot be checked by
    /// running the code on a machine without a GPU, and getting it backwards is
    /// silent — a swapped-channel texture is still a picture. So the two
    /// measured rows are pinned as literals.
    #[test]
    fn fourcc_channel_order_is_reversed_from_wgpu() {
        assert_eq!(
            wgpu_format(Fourcc::Argb8888),
            Some(wgpu::TextureFormat::Bgra8Unorm)
        );
        assert_eq!(
            wgpu_format(Fourcc::Abgr8888),
            Some(wgpu::TextureFormat::Rgba8Unorm)
        );
    }

    /// Round trip on the two proven formats. `X` variants intentionally do not
    /// round-trip back to themselves.
    #[test]
    fn fourcc_round_trips_for_the_alpha_spellings() {
        for c in [Fourcc::Argb8888, Fourcc::Abgr8888] {
            let f = wgpu_format(c).expect("mapped");
            assert_eq!(fourcc_for(f), Some(c));
            assert!(vk_format(f).is_some());
        }
    }

    /// An unmapped fourcc must be `None`, never a fallback. A "best effort"
    /// format here is the silent-wrong-pixels failure this table exists to
    /// prevent.
    #[test]
    fn unknown_fourccs_are_refused() {
        assert_eq!(wgpu_format(Fourcc::Nv12), None);
        assert_eq!(wgpu_format(Fourcc::Yuyv), None);
    }

    /// `DRM_FORMAT_MOD_INVALID` is `(1 << 56) - 1` under the vendor-0 prefix,
    /// which is easy to typo and impossible to notice: a driver handed it just
    /// refuses the image.
    #[test]
    fn invalid_modifier_is_the_drm_constant() {
        assert_eq!(INVALID, u64::from(DrmModifier::Invalid));
        assert_eq!(LINEAR, u64::from(DrmModifier::Linear));
    }
}
