//! The grader.
//!
//! Every test here needs a DRM render node and a Vulkan adapter. When there is
//! none they **skip** — printing why — rather than fail: "there is no GPU here"
//! and "the GPU got it wrong" are different answers and must not look alike.
//!
//! The pattern is a pure function of `(x, y)`, computed independently on both
//! sides, and every channel is `f32(v) / 255.0` for an integer `v`, which
//! round-trips through an 8-bit unorm format exactly. So "byte-exact" survives
//! going through a fragment shader, and an assertion failure means the pixels
//! moved wrong — not that a filter rounded.

#![cfg(all(target_os = "linux", feature = "gbm"))]

use wgpu_dmabuf::format::{Fourcc, LINEAR};
use wgpu_dmabuf::testkit::{render_node, GbmAllocator};

const W: u32 = 256;
const H: u32 = 256;

/// The known output, in **memory order for `Fourcc::Argb8888`**: B, G, R, A.
fn argb8888_pixel(x: u32, y: u32) -> [u8; 4] {
    let r = (x.wrapping_mul(3).wrapping_add(11) % 256) as u8;
    let g = (y.wrapping_mul(5).wrapping_add(29) % 256) as u8;
    let b = ((x ^ y) % 256) as u8;
    [b, g, r, 0xff]
}

/// The same pixel as the RGBA the sampling pass should write out.
fn expected_rgba(x: u32, y: u32) -> [u8; 4] {
    let [b, g, r, a] = argb8888_pixel(x, y);
    [r, g, b, a]
}

struct Ctx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    caps: wgpu_dmabuf::device::DeviceCaps,
}

/// Boot Vulkan, or explain why not. `None` means "skip".
fn ctx() -> Option<Ctx> {
    let node = render_node()?;
    eprintln!("render node: {}", node.display());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;
    eprintln!("adapter: {}", adapter.get_info().name);
    let (device, queue, caps) =
        wgpu_dmabuf::device::open_device(&adapter, &Default::default()).ok()?;
    eprintln!("caps: {caps:?}");
    Some(Ctx {
        device,
        queue,
        adapter,
        caps,
    })
}

macro_rules! gpu_or_skip {
    () => {
        match ctx() {
            Some(c) => c,
            None => {
                eprintln!("SKIP: no DRM render node / Vulkan adapter on this machine");
                return;
            }
        }
    };
}

const SAMPLE_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var v = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
    return vec4<f32>(v[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // textureLoad, not a sampler: no filtering, no coordinate rounding, so a
    // byte-exact assertion is a claim about the import and nothing else.
    return textureLoad(src, vec2<i32>(i32(pos.x), i32(pos.y)), 0);
}
"#;

/// The pattern, generated *by the shader* — so an assertion on the result also
/// proves the render pass really targeted the memory it was pointed at.
const GENERATE_WGSL: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var v = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
    return vec4<f32>(v[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);
    let r = (x * 3u + 11u) % 256u;
    let g = (y * 5u + 29u) % 256u;
    let b = (x ^ y) % 256u;
    return vec4<f32>(f32(r) / 255.0, f32(g) / 255.0, f32(b) / 255.0, 1.0);
}
"#;

fn pipeline(
    device: &wgpu::Device,
    wgsl: &str,
    target: wgpu::TextureFormat,
    bgl: Option<&wgpu::BindGroupLayout>,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(target.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Draw one full-screen triangle into `view`.
fn draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipe: &wgpu::RenderPipeline,
    bind: Option<&wgpu::BindGroup>,
    view: &wgpu::TextureView,
) {
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipe);
        if let Some(b) = bind {
            pass.set_bind_group(0, b, &[]);
        }
        pass.draw(0..3, 0..1);
    }
    queue.submit([enc.finish()]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
}

/// Sample `src` into a fresh `Rgba8Unorm` target and read the bytes back.
fn sample_and_read(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Texture) -> Vec<u8> {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("grader target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });
    let src_view = src.create_view(&Default::default());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&src_view),
        }],
    });
    let pipe = pipeline(
        device,
        SAMPLE_WGSL,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(&bgl),
    );
    let view = target.create_view(&Default::default());
    draw(device, queue, &pipe, Some(&bind), &view);
    read_texture(device, queue, &target)
}

fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<u8> {
    // 256 * 4 == 1024, already a multiple of COPY_BYTES_PER_ROW_ALIGNMENT.
    let bytes_per_row = W * 4;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(bytes_per_row * H),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let out = slice.get_mapped_range().expect("mapped range").to_vec();
    buffer.unmap();
    out
}

fn assert_matches(got: &[u8], expect: impl Fn(u32, u32) -> [u8; 4]) {
    let mut bad = 0usize;
    let mut first = None;
    for y in 0..H {
        for x in 0..W {
            let o = ((y * W + x) * 4) as usize;
            let want = expect(x, y);
            if got[o..o + 4] != want {
                bad += 1;
                first.get_or_insert((x, y, [got[o], got[o + 1], got[o + 2], got[o + 3]], want));
            }
        }
    }
    if bad != 0 {
        let (x, y, g, w) = first.unwrap();
        panic!(
            "{bad}/{} pixels wrong; first at ({x},{y}): got {g:?} want {w:?}",
            W * H
        );
    }
}

/// **The import grader.** A dma-buf allocated and filled by GBM on the CPU,
/// imported, sampled in WGSL, read back, byte-exact.
#[test]
fn import_a_gbm_dmabuf_and_sample_it_byte_exact() {
    let c = gpu_or_skip!();
    let Some(alloc) = GbmAllocator::open_default() else {
        eprintln!("SKIP: cannot open a GBM device on the render node");
        return;
    };
    let buf = alloc
        .allocate_linear(W, H, Fourcc::Argb8888, argb8888_pixel)
        .expect("gbm allocation");
    eprintln!("allocated {}", buf.dmabuf.describe());
    assert_eq!(
        buf.dmabuf.modifier, LINEAR,
        "asked for LINEAR, got something else"
    );

    let tex = wgpu_dmabuf::import(
        &c.device,
        &buf.dmabuf,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
    )
    .expect("import");
    assert_eq!(tex.format(), wgpu::TextureFormat::Bgra8Unorm);

    let got = sample_and_read(&c.device, &c.queue, &tex);
    assert_matches(&got, expected_rgba);
}

/// **The export grader.** Allocate an exportable image, render the pattern into
/// it with wgpu, then import the exported fd *back* through the public import
/// path and sample it. A wrong modifier, a wrong stride or a wrong offset all
/// fail here, and none of them would fail a `copy_texture_to_buffer` on the
/// original texture.
#[test]
fn export_a_texture_render_into_it_and_reimport_byte_exact() {
    let c = gpu_or_skip!();

    let mods = wgpu_dmabuf::shareable(
        &c.adapter,
        Fourcc::Abgr8888,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    )
    .expect("shareable modifiers");
    assert!(!mods.is_empty(), "no shareable modifier for ABGR8888");
    let values: Vec<u64> = mods.iter().map(|m| m.value).collect();
    eprintln!("shareable modifiers: {values:x?}");

    let (tex, buf) = wgpu_dmabuf::export(
        &c.device,
        &wgpu_dmabuf::ExportDescriptor {
            label: Some("exported"),
            width: W,
            height: H,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            modifiers: &values,
        },
    )
    .expect("export");
    eprintln!("exported {}", buf.describe());

    let pipe = pipeline(
        &c.device,
        GENERATE_WGSL,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
    );
    let view = tex.create_view(&Default::default());
    draw(&c.device, &c.queue, &pipe, None, &view);
    drop(view);

    let reimported = wgpu_dmabuf::import(
        &c.device,
        &buf,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
    )
    .expect("re-import the exported fd");
    let got = sample_and_read(&c.device, &c.queue, &reimported);
    assert_matches(&got, expected_rgba);
}

/// Modifier enumeration says something, and says the two halves differently.
#[test]
fn modifiers_are_enumerable_and_shareable_is_a_subset() {
    let c = gpu_or_skip!();
    let all = wgpu_dmabuf::modifiers(&c.adapter, Fourcc::Argb8888).expect("modifiers");
    assert!(!all.is_empty(), "driver reports no modifiers for ARGB8888");
    let shareable = wgpu_dmabuf::shareable(
        &c.adapter,
        Fourcc::Argb8888,
        wgpu::TextureUsages::TEXTURE_BINDING,
    )
    .expect("shareable");
    eprintln!(
        "ARGB8888: {} modifiers, {} shareable",
        all.len(),
        shareable.len()
    );
    assert!(shareable.len() <= all.len());
    for m in &shareable {
        assert!(m.is_single_plane());
        assert!(all.iter().any(|a| a.value == m.value));
    }
    assert!(
        shareable.iter().any(|m| m.value == LINEAR),
        "LINEAR must be shareable or the CPU-fill test could never work"
    );
    // An unmapped fourcc is an error, not an empty list.
    assert!(wgpu_dmabuf::modifiers(&c.adapter, Fourcc::Nv12).is_err());
}

/// **The fence grader.** arm → submit → export, then poll the fd from outside
/// Vulkan. A fence that never signals hangs a consumer, so the assertion is on
/// a bounded wait.
#[test]
fn a_completion_fence_exports_a_sync_file_that_signals() {
    let c = gpu_or_skip!();
    if !c.caps.external_semaphore_fd {
        eprintln!("SKIP: VK_KHR_external_semaphore_fd unavailable on this driver");
        return;
    }
    let fence = wgpu_dmabuf::sync::CompletionFence::new(&c.device).expect("CompletionFence");

    // Some real work, so the fence has something to be about.
    let target = c.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let pipe = pipeline(
        &c.device,
        GENERATE_WGSL,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
    );
    let view = target.create_view(&Default::default());

    let (sync_file, _) = fence
        .export_for_submit(&c.queue, || {
            let mut enc = c.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pipe);
                pass.draw(0..3, 0..1);
            }
            c.queue.submit([enc.finish()])
        })
        .expect("arm → submit → export");

    eprintln!(
        "exported sync_file provenance: {}",
        wgpu_dmabuf::sync::provenance_line(&sync_file.provenance().expect("provenance"))
    );
    assert!(
        sync_file
            .wait_timeout(std::time::Duration::from_secs(5))
            .expect("poll"),
        "the exported sync_file never signalled"
    );
    assert!(sync_file.is_signalled().expect("poll"));

    c.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("drain");
}

/// The acquire side of the handshake: a fence exported from one submit is
/// imported as a wait on the next one, and the submit still completes.
///
/// This proves the *plumbing*, not the ordering — a wait on an already-retired
/// fence orders nothing, and no timing test can tell the two apart. The
/// ordering claim rests on the Vulkan spec, not on this assertion.
#[test]
fn an_acquire_semaphore_accepts_an_exported_fence() {
    let c = gpu_or_skip!();
    if !c.caps.external_semaphore_fd {
        eprintln!("SKIP: VK_KHR_external_semaphore_fd unavailable on this driver");
        return;
    }
    let completion = wgpu_dmabuf::sync::CompletionFence::new(&c.device).expect("CompletionFence");
    let acquire = wgpu_dmabuf::sync::AcquireSemaphore::new(&c.device).expect("AcquireSemaphore");

    let (sync_file, _) = completion
        .export_for_submit(&c.queue, || {
            let enc = c.device.create_command_encoder(&Default::default());
            c.queue.submit([enc.finish()])
        })
        .expect("export");

    acquire.import(sync_file).expect("import_sync_file");
    acquire
        .wait_on_next_submit(
            &c.queue,
            wgpu_dmabuf::ash::vk::PipelineStageFlags::ALL_COMMANDS,
        )
        .expect("stage the wait");
    let enc = c.device.create_command_encoder(&Default::default());
    c.queue.submit([enc.finish()]);
    c.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("drain");

    // The wait was consumed by that submit, so nothing is left staged.
    assert!(
        !acquire.cancel_pending_wait(&c.queue).expect("cancel"),
        "the staged wait was not consumed by the submit"
    );
}

/// The producer's handshake: a render's completion fence, attached to a real
/// GBM dma-buf as its write fence, is what an implicitly synchronised reader
/// of that buffer then gets back from `EXPORT_SYNC_FILE`.
///
/// Like the acquire test this proves the plumbing — the kernel took the fence
/// and hands a fence back for readers — and prints the provenance so a run
/// shows whose fence it was. The ordering claim is the kernel's.
#[test]
fn a_completion_fence_attaches_to_a_dmabuf_as_its_write_fence() {
    let c = gpu_or_skip!();
    if !c.caps.external_semaphore_fd {
        eprintln!("SKIP: VK_KHR_external_semaphore_fd unavailable on this driver");
        return;
    }
    let Some(alloc) = GbmAllocator::open_default() else {
        eprintln!("SKIP: cannot open a GBM device on the render node");
        return;
    };
    let buf = alloc
        .allocate_linear(W, H, Fourcc::Argb8888, argb8888_pixel)
        .expect("gbm allocation");
    let fd = &buf.dmabuf.sole_plane().expect("one plane").fd;
    let completion = wgpu_dmabuf::sync::CompletionFence::new(&c.device).expect("CompletionFence");
    let (sync_file, _) = completion
        .export_for_submit(&c.queue, || {
            let enc = c.device.create_command_encoder(&Default::default());
            c.queue.submit([enc.finish()])
        })
        .expect("export");
    use std::os::fd::AsFd;
    wgpu_dmabuf::sync::import_sync_file(
        fd.as_fd(),
        sync_file.as_fd(),
        wgpu_dmabuf::sync::SyncDirection::Write,
    )
    .expect("DMA_BUF_IOCTL_IMPORT_SYNC_FILE");
    let back =
        wgpu_dmabuf::sync::export_sync_file(fd.as_fd(), wgpu_dmabuf::sync::SyncDirection::Read)
            .expect("export for a reader");
    eprintln!(
        "a reader waits on: {}",
        wgpu_dmabuf::sync::provenance_line(&back.sync_file.provenance().expect("provenance"))
    );
    assert!(
        back.sync_file
            .wait_timeout(std::time::Duration::from_secs(5))
            .expect("poll"),
        "the reader's fence never signalled"
    );
    c.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("drain");
}

/// The finding this crate's `device` module exists for: a **stock**
/// `request_device` does not enable `VK_KHR_external_semaphore_fd`, so the
/// semaphore types refuse to build on it.
///
/// If this ever starts failing, wgpu began enabling the extension and
/// [`wgpu_dmabuf::device::open_device`] can be retired.
#[test]
fn stock_request_device_cannot_fence() {
    let Some(_node) = render_node() else {
        eprintln!("SKIP: no DRM render node");
        return;
    };
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
    });
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("SKIP: no Vulkan adapter");
        return;
    };
    let Ok((device, _queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("stock"),
            required_features: adapter.features() & wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF,
            ..Default::default()
        }))
    else {
        eprintln!("SKIP: request_device failed");
        return;
    };
    match wgpu_dmabuf::sync::CompletionFence::new(&device) {
        Err(wgpu_dmabuf::Error::NoExternalSemaphoreFd) => {}
        Err(e) => panic!("unexpected error: {e}"),
        Ok(_) => panic!(
            "stock request_device now enables VK_KHR_external_semaphore_fd; \
             update the docs and consider retiring device::open_device"
        ),
    }
}

/// **The render node is the one the driver opened**, checked against the
/// kernel rather than against itself: the node's sysfs PCI vendor is the
/// adapter's vendor, and the device file carries the same `dev_t`.
#[test]
fn the_render_node_is_this_adapters() {
    let c = gpu_or_skip!();
    let info = c.adapter.get_info();
    let node = wgpu_dmabuf::device::render_node(&c.adapter).expect("render_node");
    if info.device_type == wgpu::DeviceType::Cpu {
        assert_eq!(node, None, "a software adapter has no render node");
        return;
    }
    let dev = node.expect("a hardware adapter on Mesa or NVIDIA has a render node");
    let (major, minor) = (libc::major(dev), libc::minor(dev));
    eprintln!("render node {major}:{minor} for {}", info.name);
    assert!(
        minor >= 128,
        "{major}:{minor} is a primary node, not a render node"
    );

    let path = format!("/dev/dri/renderD{minor}");
    let rdev = std::os::unix::fs::MetadataExt::rdev(&std::fs::metadata(&path).expect(&path));
    assert_eq!(rdev, dev, "{path} is not the device the driver named");

    let vendor = std::fs::read_to_string(format!("/sys/dev/char/{major}:{minor}/device/vendor"))
        .expect("sysfs vendor");
    let vendor = u32::from_str_radix(vendor.trim().trim_start_matches("0x"), 16).unwrap();
    assert_eq!(vendor, info.vendor, "the node belongs to another GPU");
}
