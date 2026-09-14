//! DRM format modifier enumeration.
//!
//! "Which modifiers can this adapter share a buffer of this format with?" is
//! the question a compositor answers when it advertises `zwp_linux_dmabuf_v1`
//! format tables, and the question an allocator answers before calling
//! `gbm_bo_create_with_modifiers2`. Vulkan answers it in two halves and both
//! are needed:
//!
//! 1. `vkGetPhysicalDeviceFormatProperties2` with
//!    `VkDrmFormatModifierPropertiesListEXT` — *which modifiers exist for this
//!    format*, with their plane counts and tiling features.
//! 2. `vkGetPhysicalDeviceImageFormatProperties2` with
//!    `VkPhysicalDeviceImageDrmFormatModifierInfoEXT` +
//!    `VkPhysicalDeviceExternalImageFormatInfo` — *and can an image with that
//!    modifier and this usage actually be imported from, or exported to, a
//!    dma-buf*.
//!
//! Half 1 alone is the trap: it lists modifiers the driver understands, not
//! modifiers it will share. [`shareable`] asks both.

use ash::vk;

use crate::error::{Error, Result};
use crate::format::{self, Fourcc};
use crate::import::vk_usage;

use wgpu::hal::api::Vulkan;

/// One DRM format modifier as the driver describes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Modifier {
    /// The modifier value, e.g. `0` for `DRM_FORMAT_MOD_LINEAR`.
    pub value: u64,
    /// How many *memory planes* an image with this modifier has. Not the same
    /// as the format's plane count: a compressed modifier can add an auxiliary
    /// plane to a single-plane format. This crate imports `plane_count == 1`
    /// only.
    pub plane_count: u32,
    /// `VkFormatFeatureFlags` for images with this modifier.
    pub features: vk::FormatFeatureFlags,
}

impl Modifier {
    /// Is this a modifier [`crate::import()`] can accept — i.e. single-plane?
    pub fn is_single_plane(&self) -> bool {
        self.plane_count == 1
    }
}

/// Every DRM format modifier the adapter knows for `fourcc`.
///
/// This is the raw driver answer: it includes multi-plane modifiers and
/// modifiers that cannot be shared as a dma-buf at all. For the list you can
/// hand an allocator, use [`shareable`].
///
/// Returns an empty vector (not an error) if the format is one the driver has
/// no modifiers for. Returns [`Error::UnmappableFourcc`] if this crate has no
/// `wgpu::TextureFormat` for the fourcc at all — the two are different answers
/// and must not look alike.
pub fn modifiers(adapter: &wgpu::Adapter, fourcc: Fourcc) -> Result<Vec<Modifier>> {
    let wgpu_format = format::wgpu_format(fourcc).ok_or(Error::UnmappableFourcc(fourcc))?;
    let vk_format = format::vk_format(wgpu_format).ok_or(Error::UnmappableFormat(wgpu_format))?;

    // SAFETY: we read the adapter's instance and physical device handles and
    // make only query calls against them.
    let hal_adapter = unsafe { adapter.as_hal::<Vulkan>() }.ok_or(Error::NotVulkan)?;
    let instance = hal_adapter.shared_instance().raw_instance();
    let pdev = hal_adapter.raw_physical_device();

    // Two-pass: ask for the count, then ask again with a buffer of that size.
    // `VkDrmFormatModifierPropertiesListEXT` is a classic "you allocate it"
    // chained struct and the first call must leave the pointer null.
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut props = vk::FormatProperties2::default().push_next(&mut list);
    // SAFETY: `props` chains one live, correctly typed struct.
    unsafe { instance.get_physical_device_format_properties2(pdev, vk_format, &mut props) };
    let count = list.drm_format_modifier_count as usize;
    if count == 0 {
        return Ok(Vec::new());
    }

    let mut storage = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
    let mut list = vk::DrmFormatModifierPropertiesListEXT {
        drm_format_modifier_count: count as u32,
        p_drm_format_modifier_properties: storage.as_mut_ptr(),
        ..Default::default()
    };
    let mut props = vk::FormatProperties2::default().push_next(&mut list);
    // SAFETY: `storage` holds exactly `count` entries of the layout the driver
    // writes and outlives the call.
    unsafe { instance.get_physical_device_format_properties2(pdev, vk_format, &mut props) };

    Ok(storage
        .iter()
        .take(list.drm_format_modifier_count as usize)
        .map(|m| Modifier {
            value: m.drm_format_modifier,
            plane_count: m.drm_format_modifier_plane_count,
            features: m.drm_format_modifier_tiling_features,
        })
        .collect())
}

/// The single-plane modifiers for `fourcc` that this adapter can both **import
/// from** and **export to** a dma-buf, with `usage`.
///
/// This is the list to hand `gbm_bo_create_with_modifiers2`, and the list a
/// compositor should advertise. Both directions are required rather than just
/// the one the caller needs today: a modifier that is importable but not
/// exportable makes a buffer that can never come back, and discovering that at
/// the far end of a handoff is expensive.
///
/// Multi-plane modifiers are filtered out because [`crate::import()`] cannot
/// accept them.
pub fn shareable(
    adapter: &wgpu::Adapter,
    fourcc: Fourcc,
    usage: wgpu::TextureUsages,
) -> Result<Vec<Modifier>> {
    let wgpu_format = format::wgpu_format(fourcc).ok_or(Error::UnmappableFourcc(fourcc))?;
    let vk_format = format::vk_format(wgpu_format).ok_or(Error::UnmappableFormat(wgpu_format))?;
    let vk_usage = vk_usage(usage);
    let all = modifiers(adapter, fourcc)?;

    // SAFETY: query calls only, as above.
    let hal_adapter = unsafe { adapter.as_hal::<Vulkan>() }.ok_or(Error::NotVulkan)?;
    let instance = hal_adapter.shared_instance().raw_instance();
    let pdev = hal_adapter.raw_physical_device();

    let mut out = Vec::new();
    for m in all.into_iter().filter(Modifier::is_single_plane) {
        let mut mod_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(m.value)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut ext_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(vk_format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk_usage)
            .push_next(&mut mod_info)
            .push_next(&mut ext_info);
        let mut ext_props = vk::ExternalImageFormatProperties::default();
        let mut props = vk::ImageFormatProperties2::default().push_next(&mut ext_props);
        // SAFETY: one live info chain, one live output chain. A driver that
        // does not support the combination returns FORMAT_NOT_SUPPORTED, which
        // is an answer, not a failure.
        let supported = unsafe {
            instance.get_physical_device_image_format_properties2(pdev, &info, &mut props)
        }
        .is_ok();
        if !supported {
            continue;
        }
        let f = ext_props
            .external_memory_properties
            .external_memory_features;
        if f.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            && f.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
        {
            out.push(m);
        }
    }

    Ok(out)
}

/// Pick a modifier to allocate with from a shareable list.
///
/// Prefers the highest non-linear modifier — tiled layouts are what the GPU
/// actually wants — and falls back to `LINEAR` only if nothing else is
/// shareable. A CPU writer (`gbm_bo_map`) must **not** use this: it needs
/// [`crate::format::LINEAR`] specifically, because a tiled buffer's bytes are
/// not in raster order and filling one row-by-row produces a swizzled picture.
pub fn best_for_gpu(list: &[Modifier]) -> Option<u64> {
    list.iter()
        .map(|m| m.value)
        .filter(|v| *v != format::LINEAR)
        .max()
        .or_else(|| list.iter().map(|m| m.value).find(|v| *v == format::LINEAR))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `best_for_gpu` must never return LINEAR while a tiled modifier is on
    /// offer, and must never return `None` from a non-empty list.
    #[test]
    fn best_prefers_tiled_but_accepts_linear() {
        let tiled = Modifier {
            value: 0x0100_0000_0000_0001,
            plane_count: 1,
            features: vk::FormatFeatureFlags::empty(),
        };
        let linear = Modifier {
            value: format::LINEAR,
            plane_count: 1,
            features: vk::FormatFeatureFlags::empty(),
        };
        assert_eq!(best_for_gpu(&[linear, tiled]), Some(tiled.value));
        assert_eq!(best_for_gpu(&[linear]), Some(format::LINEAR));
        assert_eq!(best_for_gpu(&[]), None);
    }
}
