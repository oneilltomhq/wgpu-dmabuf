//! Opening a `wgpu::Device` that can actually do all of this.
//!
//! Stock `wgpu::Adapter::request_device` is enough for the **memory** half:
//! wgpu-hal 30 already enables `VK_KHR_external_memory_fd`,
//! `VK_EXT_external_memory_dma_buf` and `VK_EXT_image_drm_format_modifier`
//! whenever the driver has them, and exposes that as
//! [`wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF`]. Import and export need
//! no patch, no fork and no hand-rolled `vkCreateDevice`.
//!
//! It is not enough for the **fence** half. `VK_KHR_external_semaphore_fd`
//! appears nowhere in wgpu-hal 30.0.1, so `vkImportSemaphoreFdKHR` /
//! `vkGetSemaphoreFdKHR` are not callable on a device wgpu opened — even though
//! the `Queue::add_wait_semaphore` / `add_signal_semaphore` seam
//! [`crate::sync`] hangs off *is* upstream.
//!
//! [`open_device`] closes that gap the only way the wgpu API allows: call
//! `vkCreateDevice` ourselves with the extra extension on top of wgpu's own
//! list, then adopt the result through `Adapter::device_from_raw` +
//! `Adapter::create_device_from_hal`. wgpu takes ownership of the `VkDevice`
//! and destroys it normally; nothing is leaked and nothing is patched.

use std::ffi::CStr;

use ash::vk;

use crate::error::{Error, Result};

use wgpu::hal::api::Vulkan;

/// What to open.
#[derive(Debug, Clone)]
pub struct DeviceOptions<'a> {
    /// Debug label for the device.
    pub label: Option<&'a str>,
    /// wgpu features *in addition to* [`wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF`],
    /// which is always requested when the adapter offers it.
    pub required_features: wgpu::Features,
    /// Limits to request, as for `wgpu::DeviceDescriptor`.
    pub required_limits: wgpu::Limits,
    /// Allocator hints, as for `wgpu::DeviceDescriptor`.
    pub memory_hints: wgpu::MemoryHints,
    /// Vulkan device extensions to enable on top of wgpu's own list and
    /// `VK_KHR_external_semaphore_fd`. An extension the driver does not expose
    /// is skipped rather than fatal; check [`DeviceCaps`] afterwards.
    pub extra_extensions: &'a [&'static CStr],
}

impl Default for DeviceOptions<'_> {
    fn default() -> Self {
        Self {
            label: Some("wgpu-dmabuf"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            extra_extensions: &[],
        }
    }
}

/// What the opened device actually got, as opposed to what was asked for.
///
/// Reported rather than asserted: a caller that only imports does not need the
/// semaphore extension, and this crate has no business deciding that for them.
/// A caller that *does* need it should check here, at boot, rather than
/// discover it from [`Error::NoExternalSemaphoreFd`] mid-frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCaps {
    /// dma-buf import and export are available.
    pub dmabuf: bool,
    /// `VK_KHR_external_semaphore_fd` is enabled, so [`crate::sync`]'s
    /// semaphore types will work.
    pub external_semaphore_fd: bool,
}

/// Open a Vulkan device with dma-buf interop *and* `sync_file` fencing enabled.
///
/// Returns the device, its queue, and [`DeviceCaps`] saying what was actually
/// granted.
///
/// # Why not just `request_device`?
///
/// Because `request_device` cannot be told to enable an extension wgpu has no
/// feature bit for. See the module docs.
///
/// # Safety of the adoption
///
/// The `VkDevice` is created here and handed to wgpu with no drop callback,
/// which means **wgpu owns it** and destroys it when the `wgpu::Device` is
/// dropped. Do not keep an `ash::Device` clone alive past that point expecting
/// it to work; the dispatch table survives, the device does not.
pub fn open_device(
    adapter: &wgpu::Adapter,
    options: &DeviceOptions<'_>,
) -> Result<(wgpu::Device, wgpu::Queue, DeviceCaps)> {
    let features = options.required_features
        | (adapter.features() & wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF);

    // SAFETY: we read the adapter's handles, create a device from them, and
    // hand that device straight back to wgpu. Nothing wgpu owns is destroyed.
    let (open_device, caps) = unsafe {
        let hal_adapter = adapter.as_hal::<Vulkan>().ok_or(Error::NotVulkan)?;
        let instance = hal_adapter.shared_instance().raw_instance().clone();
        let pdev = hal_adapter.raw_physical_device();

        let available = instance
            .enumerate_device_extension_properties(pdev)
            .map_err(Error::vk("vkEnumerateDeviceExtensionProperties"))?;
        let has = |name: &CStr| {
            available
                .iter()
                .any(|p| p.extension_name_as_c_str() == Ok(name))
        };

        let mut extensions = hal_adapter.required_device_extensions(features);
        let mut wanted: Vec<&'static CStr> = vec![ash::khr::external_semaphore_fd::NAME];
        wanted.extend_from_slice(options.extra_extensions);
        for name in wanted {
            if has(name) && !extensions.contains(&name) {
                extensions.push(name);
            }
        }
        let caps = DeviceCaps {
            dmabuf: features.contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF),
            external_semaphore_fd: extensions.contains(&ash::khr::external_semaphore_fd::NAME),
        };

        let mut phd_features = hal_adapter.physical_device_features(&extensions, features);

        let family_index = instance
            .get_physical_device_queue_family_properties(pdev)
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or(Error::Hal("no graphics queue family".into()))?
            as u32;

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family_index)
            .queue_priorities(&priorities)];
        let name_ptrs: Vec<*const core::ffi::c_char> =
            extensions.iter().map(|c| c.as_ptr()).collect();
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&name_ptrs);
        let create = phd_features.add_to_device_create(create);
        let raw_device = instance
            .create_device(pdev, &create, None)
            .map_err(Error::vk("vkCreateDevice"))?;

        // `None` drop callback: wgpu takes ownership of the VkDevice.
        let open = hal_adapter
            .device_from_raw(
                raw_device,
                None,
                &extensions,
                features,
                &options.required_limits,
                &options.memory_hints,
                family_index,
                0,
            )
            .map_err(|e| Error::Hal(format!("device_from_raw: {e:?}")))?;
        (open, caps)
    };

    let desc = wgpu::DeviceDescriptor {
        label: options.label,
        required_features: features,
        required_limits: options.required_limits.clone(),
        memory_hints: options.memory_hints.clone(),
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    };
    // SAFETY: `open_device` was created from this very adapter, with the
    // extensions and features named in `desc`.
    let (device, queue) = unsafe { adapter.create_device_from_hal::<Vulkan>(open_device, &desc) }
        .map_err(|e| Error::RequestDevice(e.to_string()))?;

    Ok((device, queue, caps))
}

/// The raw `VkDevice` handle behind a `wgpu::Device`, as an opaque `u64`.
///
/// "Two distinct devices" has to be an *identity* assertion, not an inference
/// from two different `Arc` addresses: an `Arc` is a wgpu-level fact, the
/// `VkDevice` is the thing the driver schedules. Reads a handle; creates,
/// destroys and mutates nothing.
pub fn vk_device_handle(device: &wgpu::Device) -> Option<u64> {
    use ash::vk::Handle as _;
    // SAFETY: reads the hal device's handle and nothing else.
    unsafe {
        device
            .as_hal::<Vulkan>()
            .map(|hal| hal.raw_device().handle().as_raw())
    }
}
