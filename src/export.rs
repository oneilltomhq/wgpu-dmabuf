//! `wgpu::Texture` → dma-buf.
//!
//! wgpu has no export helper of its own — `wgpu_hal::vulkan::Device` ships
//! `texture_from_dmabuf_fd` for the import direction and nothing for this one —
//! so the image is created by hand against wgpu's own `VkDevice` with `ash`,
//! then adopted with `texture_from_raw(.., TextureMemory::External)`.
//!
//! The result is an ordinary `wgpu::Texture` you can render into, whose memory
//! is a dma-buf another device or another process can import.

use std::os::fd::{FromRawFd, OwnedFd};

use ash::vk;

use crate::dmabuf::Dmabuf;
use crate::error::{Error, Result};
use crate::format;
use crate::import::vk_usage;

use wgpu::hal::api::Vulkan;

/// What to allocate.
#[derive(Debug, Clone)]
pub struct ExportDescriptor<'a> {
    /// Debug label for the texture.
    pub label: Option<&'a str>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Must be a format [`crate::format::fourcc_for`] maps.
    pub format: wgpu::TextureFormat,
    /// wgpu-level usage for the returned texture. `RENDER_ATTACHMENT` is the
    /// point of the exercise; `COPY_SRC` is useful for grading it.
    pub usage: wgpu::TextureUsages,
    /// Candidate DRM format modifiers, in no particular order — the driver
    /// picks one and the choice is reported in the returned [`Dmabuf`].
    ///
    /// Get this list from [`crate::modifiers::shareable`]. Passing a single
    /// `LINEAR` is legal and is what a CPU reader on the far end needs.
    pub modifiers: &'a [u64],
}

/// Allocate an exportable image, adopt it as a `wgpu::Texture`, and hand back
/// its dma-buf.
///
/// The returned texture's initial tracker state is
/// `TextureUses::UNINITIALIZED`: the image was just created in
/// `VK_IMAGE_LAYOUT_UNDEFINED` and has no contents, so telling wgpu anything
/// else would be a lie it would act on.
///
/// # Lifetime and ownership
///
/// The `VkImage` and its `VkDeviceMemory` are owned by the returned texture and
/// destroyed when it is dropped — **not** when the exported fd is closed. The
/// fd in the returned [`Dmabuf`] is an independent reference to the same
/// memory, so the memory itself survives as long as *either* side holds it, but
/// the `VkImage` view of it does not. Consumers must therefore drop the texture
/// only once the GPU is idle; a `wgpu::Device::poll(PollType::wait())` before
/// teardown is the cheap way to guarantee that.
///
/// # Synchronisation
///
/// As with [`crate::import()`], this establishes sharing and not ordering. The
/// far end sees whatever is in the memory when it looks. Pair it with
/// [`crate::sync::CompletionFence`].
pub fn export(
    device: &wgpu::Device,
    desc: &ExportDescriptor<'_>,
) -> Result<(wgpu::Texture, Dmabuf)> {
    let fourcc = format::fourcc_for(desc.format).ok_or(Error::UnmappableFormat(desc.format))?;
    let vk_format = format::vk_format(desc.format).ok_or(Error::UnmappableFormat(desc.format))?;
    if desc.modifiers.is_empty() {
        return Err(Error::NoShareableModifier { fourcc });
    }
    if !device
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return Err(Error::MissingDmabufFeature);
    }

    // SAFETY: we read handles out of the hal device; everything created below
    // is created by us and owned by us.
    let (ash_device, instance, pdev) = unsafe {
        let hal = device.as_hal::<Vulkan>().ok_or(Error::NotVulkan)?;
        (
            hal.raw_device().clone(),
            hal.shared_instance().raw_instance().clone(),
            hal.raw_physical_device(),
        )
    };
    let ext_memory_fd = ash::khr::external_memory_fd::Device::new(&instance, &ash_device);
    let ext_modifier = ash::ext::image_drm_format_modifier::Device::new(&instance, &ash_device);

    let usage = vk_usage(desc.usage);
    let extent = vk::Extent3D {
        width: desc.width,
        height: desc.height,
        depth: 1,
    };

    let mut mod_list =
        vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(desc.modifiers);
    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut mod_list)
        .push_next(&mut external);

    // SAFETY: a well-formed create-info chain on a live device.
    let image = unsafe { ash_device.create_image(&image_info, None) }
        .map_err(Error::vk("vkCreateImage(exportable)"))?;

    // Everything after this point must destroy `image` on the way out.
    let built = (|| -> Result<(vk::DeviceMemory, Dmabuf)> {
        // SAFETY: `image` is live and was created by us.
        let reqs = unsafe { ash_device.get_image_memory_requirements(image) };
        // SAFETY: a plain query on the physical device.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pdev) };
        let type_index = (0..mem_props.memory_type_count)
            .find(|i| {
                reqs.memory_type_bits & (1 << i) != 0
                    && mem_props.memory_types[*i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or(Error::NoCompatibleMemoryType)?;

        // A dedicated allocation is not an optimisation here: an exported
        // dma-buf must be the whole of its `VkDeviceMemory`, because the fd
        // names the allocation and the importer has no way to be told about a
        // sub-allocation offset.
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut export_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index)
            .push_next(&mut dedicated)
            .push_next(&mut export_info);
        // SAFETY: well-formed allocate-info chain on a live device.
        let memory = unsafe { ash_device.allocate_memory(&alloc, None) }
            .map_err(Error::vk("vkAllocateMemory(exportable)"))?;

        let bound = (|| -> Result<Dmabuf> {
            // SAFETY: freshly allocated memory sized to the image's requirements.
            unsafe { ash_device.bind_image_memory(image, memory, 0) }
                .map_err(Error::vk("vkBindImageMemory"))?;

            // Which modifier did the driver actually choose from the list? This
            // has to be asked, not assumed: the caller offered several and the
            // importer needs the one that was used.
            let mut chosen = vk::ImageDrmFormatModifierPropertiesEXT::default();
            // SAFETY: `image` is a bound DRM-modifier image on this device.
            unsafe { ext_modifier.get_image_drm_format_modifier_properties(image, &mut chosen) }
                .map_err(Error::vk("vkGetImageDrmFormatModifierPropertiesEXT"))?;

            // SAFETY: memory plane 0 exists for every single-plane modifier.
            let layout = unsafe {
                ash_device.get_image_subresource_layout(
                    image,
                    vk::ImageSubresource {
                        aspect_mask: vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                        mip_level: 0,
                        array_layer: 0,
                    },
                )
            };

            let get_fd = vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            // SAFETY: `memory` was allocated with DMA_BUF as an export handle type.
            let raw_fd = unsafe { ext_memory_fd.get_memory_fd(&get_fd) }
                .map_err(Error::vk("vkGetMemoryFdKHR"))?;
            // SAFETY: the driver just transferred ownership of this fd to us.
            let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

            Ok(Dmabuf::single_plane(
                desc.width,
                desc.height,
                fourcc,
                chosen.drm_format_modifier,
                fd,
                layout.offset as u32,
                layout.row_pitch as u32,
            ))
        })();

        match bound {
            Ok(buf) => Ok((memory, buf)),
            Err(e) => {
                // SAFETY: allocated by us above and not referenced by anything.
                unsafe { ash_device.free_memory(memory, None) };
                Err(e)
            }
        }
    })();

    let (memory, buf) = match built {
        Ok(v) => v,
        Err(e) => {
            // SAFETY: created by us above and not referenced by anything.
            unsafe { ash_device.destroy_image(image, None) };
            return Err(e);
        }
    };

    let size = wgpu::Extent3d {
        width: desc.width,
        height: desc.height,
        depth_or_array_layers: 1,
    };
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: desc.label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: desc.format,
        usage: crate::import::hal_uses(desc.usage),
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    // `TextureMemory::External` plus a drop callback: wgpu-hal must not try to
    // free memory it did not allocate, and the callback is where the image and
    // its memory are destroyed once wgpu is done with the texture.
    let cleanup = {
        let ash_device = ash_device.clone();
        Box::new(move || {
            // SAFETY: both handles were created here and wgpu has just
            // relinquished the texture wrapping them. The caller's contract
            // (see the doc comment) is that the queue is idle.
            unsafe {
                ash_device.destroy_image(image, None);
                ash_device.free_memory(memory, None);
            }
        }) as wgpu::hal::DropCallback
    };

    // SAFETY: `image` was created respecting `hal_desc`, `view_formats` is
    // empty (the image has no MUTABLE_FORMAT flag), and the memory outlives the
    // texture because the drop callback is what frees it.
    let hal_texture = unsafe {
        let hal = device.as_hal::<Vulkan>().ok_or(Error::NotVulkan)?;
        hal.texture_from_raw(
            image,
            &hal_desc,
            Some(cleanup),
            wgpu::hal::vulkan::TextureMemory::External,
        )
    };

    let tex_desc = wgpu::TextureDescriptor {
        label: desc.label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: desc.format,
        usage: desc.usage,
        view_formats: &[],
    };
    // SAFETY: `hal_texture` matches `tex_desc` and the image really is in the
    // undefined/uninitialised state we declare.
    let texture = unsafe {
        device.create_texture_from_hal::<Vulkan>(
            hal_texture,
            &tex_desc,
            wgpu::TextureUses::UNINITIALIZED,
        )
    };

    Ok((texture, buf))
}
