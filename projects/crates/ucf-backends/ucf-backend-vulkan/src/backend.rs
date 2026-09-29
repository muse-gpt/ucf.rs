use std::collections::BTreeMap;

use ash::vk;
use ash::{Device, Entry, Instance};
use ucf_capability::{
    pick_descriptor_strategy, pick_pipeline_strategy, pick_sync_strategy, DescriptorStrategy,
    Feature, FeatureSet, PipelineStrategy, SyncStrategy,
};
use ucf_emitter::{emit_raster_tri_spirv, emit_spirv, program_from_task};
use ucf_ir::{Graph, TaskKind, TaskNode};
use ucf_scheduler::{Backend, Error as SchedulerError, Result};
use ucf_types::ResourceId;

use crate::params::{f32_param, resource_param, str_param, u32_param, BackendError};

struct DeviceBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: usize,
}

/// Resolved degrade-chain strategies wired into this backend instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WiredStrategies {
    /// Descriptor binding path chosen from [`FeatureSet`].
    pub descriptor: DescriptorStrategy,
    /// Pipeline object path chosen from [`FeatureSet`].
    pub pipeline: PipelineStrategy,
    /// Sync path chosen from [`FeatureSet`].
    pub sync: SyncStrategy,
}

/// Vulkan compute / graphics backend with device-resident buffers.
pub struct VulkanBackend {
    _entry: Entry,
    instance: Instance,
    physical: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    /// Queue family supports `GRAPHICS` (required for Raster `draw=tri`).
    graphics: bool,
    buffers: BTreeMap<ResourceId, DeviceBuffer>,
    strategies: WiredStrategies,
    /// Last descriptor path label taken by a submit (for thin-gate assertions).
    last_descriptor_path: &'static str,
    /// Last sync path label taken by a submit (for thin-gate assertions).
    last_sync_path: &'static str,
    /// Last pipeline path label taken by a submit (for thin-gate assertions).
    last_pipeline_path: &'static str,
}

impl VulkanBackend {
    /// Open the first physical device with a compute-capable queue.
    pub fn new() -> Result<Self> {
        unsafe {
            let entry = Entry::load().map_err(|e| map_backend_err(BackendError(e.to_string())))?;
            let app = vk::ApplicationInfo::default()
                .application_name(c"ucf")
                .application_version(0)
                .engine_name(c"ucf")
                .engine_version(0)
                .api_version(vk::API_VERSION_1_1);
            let info = vk::InstanceCreateInfo::default().application_info(&app);
            let instance = entry
                .create_instance(&info, None)
                .map_err(|e| map_backend_err(BackendError(format!("create_instance: {e}"))))?;

            let physicals = instance
                .enumerate_physical_devices()
                .map_err(|e| map_backend_err(BackendError(format!("enumerate devices: {e}"))))?;
            let (physical, queue_family, graphics) = physicals
                .iter()
                .find_map(|&pdev| {
                    let props = instance.get_physical_device_queue_family_properties(pdev);
                    props.iter().enumerate().find_map(|(i, q)| {
                        if q.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                            Some((pdev, i as u32, true))
                        } else {
                            None
                        }
                    })
                })
                .or_else(|| {
                    physicals.iter().find_map(|&pdev| {
                        let props = instance.get_physical_device_queue_family_properties(pdev);
                        props.iter().enumerate().find_map(|(i, q)| {
                            if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                                Some((pdev, i as u32, false))
                            } else {
                                None
                            }
                        })
                    })
                })
                .ok_or_else(|| {
                    map_backend_err(BackendError("no Vulkan compute/graphics queue".into()))
                })?;

            let priorities = [1.0f32];
            let qinfo = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities);
            let dinfo = vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&qinfo));
            let device = instance
                .create_device(physical, &dinfo, None)
                .map_err(|e| map_backend_err(BackendError(format!("create_device: {e}"))))?;
            let queue = device.get_device_queue(queue_family, 0);
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = device
                .create_command_pool(&pool_info, None)
                .map_err(|e| map_backend_err(BackendError(format!("command_pool: {e}"))))?;

            let features = FeatureSet::new()
                .with(Feature::Bindless)
                .with(Feature::DescriptorBuffer)
                .with(Feature::DynamicRendering)
                .with(Feature::Synchronization2);
            let strategies = WiredStrategies {
                descriptor: pick_descriptor_strategy(&features),
                pipeline: pick_pipeline_strategy(&features),
                sync: pick_sync_strategy(&features),
            };

            Ok(Self {
                _entry: entry,
                instance,
                physical,
                device,
                queue,
                command_pool,
                graphics,
                buffers: BTreeMap::new(),
                strategies,
                last_descriptor_path: "",
                last_sync_path: "",
                last_pipeline_path: "",
            })
        }
    }

    /// Degrade-chain strategies resolved when the backend opened.
    pub fn strategies(&self) -> WiredStrategies {
        self.strategies
    }

    /// Descriptor path label from the last submit that bound descriptors.
    pub fn last_descriptor_path(&self) -> &'static str {
        self.last_descriptor_path
    }

    /// Sync path label from the last submit that issued a barrier.
    pub fn last_sync_path(&self) -> &'static str {
        self.last_sync_path
    }

    /// Pipeline path label from the last submit that created a pipeline.
    pub fn last_pipeline_path(&self) -> &'static str {
        self.last_pipeline_path
    }

    fn mark_sync_path(&mut self) {
        self.last_sync_path = match self.strategies.sync {
            SyncStrategy::EnhancedBarriers => "enhanced_barriers",
            SyncStrategy::Synchronization2 => "synchronization2",
            SyncStrategy::LegacyBarriers => "legacy_barriers",
        };
    }

    fn mark_descriptor_path(&mut self) {
        self.last_descriptor_path = match self.strategies.descriptor {
            DescriptorStrategy::ResourceDescriptorHeap => "resource_descriptor_heap",
            DescriptorStrategy::DescriptorBuffer => "descriptor_buffer",
            DescriptorStrategy::BindlessIndexing => "bindless_indexing",
            DescriptorStrategy::TraditionalSets => "traditional_sets",
        };
    }

    fn mark_pipeline_path(&mut self) {
        self.last_pipeline_path = match self.strategies.pipeline {
            PipelineStrategy::ShaderObject => "shader_object",
            PipelineStrategy::PipelineLibrary => "pipeline_library",
            PipelineStrategy::PsoPrecache => "pso_precache",
        };
    }

    /// Upload host `f32` values into an allocated device buffer.
    pub fn write_f32(&mut self, id: ResourceId, values: &[f32]) -> Result<()> {
        let bytes = values.len() * 4;
        let buf = self.buffers.get(&id).ok_or_else(|| {
            map_backend_err(BackendError(format!("resource {} not allocated", id.0)))
        })?;
        if buf.bytes != bytes {
            return Err(map_backend_err(BackendError(format!(
                "resource {} expected {} bytes, got {bytes}",
                id.0, buf.bytes
            ))));
        }
        let mut host = Vec::with_capacity(bytes);
        for v in values {
            host.extend_from_slice(&v.to_le_bytes());
        }
        self.upload(id, &host)
    }

    /// Download a device buffer as `f32` values.
    pub fn read_f32(&mut self, id: ResourceId) -> Result<Vec<f32>> {
        let bytes = self
            .buffers
            .get(&id)
            .ok_or_else(|| {
                map_backend_err(BackendError(format!("resource {} not allocated", id.0)))
            })?
            .bytes;
        let host = self.readback(id, bytes)?;
        Ok(host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Download a device buffer as raw bytes (e.g. `R8G8B8A8` raster readback).
    pub fn read_u8(&mut self, id: ResourceId) -> Result<Vec<u8>> {
        let bytes = self
            .buffers
            .get(&id)
            .ok_or_else(|| {
                map_backend_err(BackendError(format!("resource {} not allocated", id.0)))
            })?
            .bytes;
        self.readback(id, bytes)
    }

    /// Run every task on this backend. Caller must [`prepare`] and seed inputs first.
    pub fn run_prepared(&mut self, graph: &Graph) -> Result<()> {
        graph.validate().map_err(SchedulerError::from)?;
        let order = graph
            .tasks
            .topological_order()
            .map_err(SchedulerError::from)?;
        for task_id in order {
            let task = graph
                .tasks
                .nodes
                .iter()
                .find(|t| t.id == task_id)
                .expect("task in order must exist");
            self.submit_task(graph, task)?;
        }
        Ok(())
    }

    fn upload(&mut self, id: ResourceId, bytes: &[u8]) -> Result<()> {
        unsafe {
            let (staging, staging_mem) = self.create_buffer(
                bytes.len(),
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            {
                let ptr = self
                    .device
                    .map_memory(staging_mem, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())
                    .map_err(vk_err)?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
                self.device.unmap_memory(staging_mem);
            }
            let dst = self.buffers.get(&id).unwrap().buffer;
            self.with_commands(|cmd| {
                let region = vk::BufferCopy::default().size(bytes.len() as u64);
                self.device
                    .cmd_copy_buffer(cmd, staging, dst, std::slice::from_ref(&region));
            })?;
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(staging_mem, None);
            Ok(())
        }
    }

    fn readback(&mut self, id: ResourceId, bytes: usize) -> Result<Vec<u8>> {
        unsafe {
            let (staging, staging_mem) = self.create_buffer(
                bytes,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let src = self.buffers.get(&id).unwrap().buffer;
            self.with_commands(|cmd| {
                let region = vk::BufferCopy::default().size(bytes as u64);
                self.device
                    .cmd_copy_buffer(cmd, src, staging, std::slice::from_ref(&region));
            })?;
            let mut out = vec![0u8; bytes];
            let ptr = self
                .device
                .map_memory(staging_mem, 0, bytes as u64, vk::MemoryMapFlags::empty())
                .map_err(vk_err)?;
            std::ptr::copy_nonoverlapping(ptr.cast(), out.as_mut_ptr(), bytes);
            self.device.unmap_memory(staging_mem);
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(staging_mem, None);
            Ok(out)
        }
    }

    fn create_buffer(
        &self,
        bytes: usize,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(bytes.max(1) as u64)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).map_err(vk_err)?;
            let reqs = self.device.get_buffer_memory_requirements(buffer);
            let memory_type = self
                .find_memory_type(reqs.memory_type_bits, props)
                .ok_or_else(|| map_backend_err(BackendError("no suitable memory type".into())))?;
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(memory_type);
            let memory = self.device.allocate_memory(&alloc, None).map_err(vk_err)?;
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(vk_err)?;
            Ok((buffer, memory))
        }
    }

    fn create_image(
        &self,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory)> {
        unsafe {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = self.device.create_image(&info, None).map_err(vk_err)?;
            let reqs = self.device.get_image_memory_requirements(image);
            let memory_type = self
                .find_memory_type(reqs.memory_type_bits, props)
                .ok_or_else(|| {
                    map_backend_err(BackendError("no suitable image memory type".into()))
                })?;
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(memory_type);
            let memory = self.device.allocate_memory(&alloc, None).map_err(vk_err)?;
            self.device
                .bind_image_memory(image, memory, 0)
                .map_err(vk_err)?;
            Ok((image, memory))
        }
    }

    fn find_memory_type(&self, type_bits: u32, props: vk::MemoryPropertyFlags) -> Option<u32> {
        unsafe {
            let mem = self
                .instance
                .get_physical_device_memory_properties(self.physical);
            (0..mem.memory_type_count).find(|&i| {
                (type_bits & (1 << i)) != 0
                    && mem.memory_types[i as usize]
                        .property_flags
                        .contains(props)
            })
        }
    }

    fn with_commands(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        unsafe {
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmds = self
                .device
                .allocate_command_buffers(&alloc)
                .map_err(vk_err)?;
            let cmd = cmds[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin).map_err(vk_err)?;
            record(cmd);
            self.device.end_command_buffer(cmd).map_err(vk_err)?;
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit), vk::Fence::null())
                .map_err(vk_err)?;
            self.device.queue_wait_idle(self.queue).map_err(vk_err)?;
            self.device
                .free_command_buffers(self.command_pool, &cmds);
            Ok(())
        }
    }

    fn run_copy(&mut self, task: &TaskNode) -> Result<()> {
        let src = resource_param(task, "src").map_err(map_backend_err)?;
        let dst = resource_param(task, "dst").map_err(map_backend_err)?;
        let src_bytes = self.buf_bytes(src)?;
        let dst_bytes = self.buf_bytes(dst)?;
        if src_bytes != dst_bytes {
            return Err(map_backend_err(BackendError(format!(
                "copy size mismatch: src {src_bytes} dst {dst_bytes}"
            ))));
        }
        let src_buf = self.buffers.get(&src).unwrap().buffer;
        let dst_buf = self.buffers.get(&dst).unwrap().buffer;
        self.with_commands(|cmd| unsafe {
            let region = vk::BufferCopy::default().size(src_bytes as u64);
            self.device
                .cmd_copy_buffer(cmd, src_buf, dst_buf, std::slice::from_ref(&region));
        })
    }

    fn run_fill(&mut self, task: &TaskNode) -> Result<()> {
        let dst = resource_param(task, "dst").map_err(map_backend_err)?;
        let _value = f32_param(task, "value").map_err(map_backend_err)?;
        let bytes = self.buf_bytes(dst)?;
        if bytes % 4 != 0 {
            return Err(map_backend_err(BackendError(format!(
                "fill destination {} length {bytes} is not a multiple of 4",
                dst.0
            ))));
        }
        let count = (bytes / 4) as u32;
        let program = program_from_task(task);
        let spirv = emit_spirv(&program).map_err(map_emit_err)?;
        let dst_buf = self.buffers.get(&dst).unwrap().buffer;
        self.dispatch_compute(
            &spirv,
            &program.entry,
            &[dst_buf],
            &count.to_ne_bytes(),
            count.div_ceil(64).max(1),
        )
    }

    fn run_matmul(&mut self, task: &TaskNode) -> Result<()> {
        let a = resource_param(task, "a").map_err(map_backend_err)?;
        let b = resource_param(task, "b").map_err(map_backend_err)?;
        let out = resource_param(task, "out").map_err(map_backend_err)?;
        let m = u32_param(task, "m").map_err(map_backend_err)?;
        let n = u32_param(task, "n").map_err(map_backend_err)?;
        let k = u32_param(task, "k").map_err(map_backend_err)?;
        expect_bytes(a, self.buf_bytes(a)?, (m as usize) * (k as usize) * 4)?;
        expect_bytes(b, self.buf_bytes(b)?, (k as usize) * (n as usize) * 4)?;
        expect_bytes(out, self.buf_bytes(out)?, (m as usize) * (n as usize) * 4)?;

        let program = program_from_task(task);
        let spirv = emit_spirv(&program).map_err(map_emit_err)?;
        let a_buf = self.buffers.get(&a).unwrap().buffer;
        let b_buf = self.buffers.get(&b).unwrap().buffer;
        let out_buf = self.buffers.get(&out).unwrap().buffer;
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&m.to_ne_bytes());
        push[4..8].copy_from_slice(&n.to_ne_bytes());
        push[8..12].copy_from_slice(&k.to_ne_bytes());
        self.dispatch_compute(
            &spirv,
            &program.entry,
            &[a_buf, b_buf, out_buf],
            &push,
            (m * n).div_ceil(64).max(1),
        )
    }

    /// Rasterize into an R8G8B8A8 image and copy packed pixels into buffer `dst`.
    ///
    /// Params: `dst`, `width`, `height`, optional `r`/`g`/`b`/`a` in `0..1`,
    /// optional `draw` = `clear` (default) or `tri` (yellow NDC triangle after clear).
    fn run_raster(&mut self, task: &TaskNode) -> Result<()> {
        let dst = resource_param(task, "dst").map_err(map_backend_err)?;
        let width = u32_param(task, "width").map_err(map_backend_err)?;
        let height = u32_param(task, "height").map_err(map_backend_err)?;
        if width == 0 || height == 0 {
            return Err(map_backend_err(BackendError(
                "raster width/height must be > 0".into(),
            )));
        }
        let draw = str_param(task, "draw").unwrap_or("clear");
        if draw != "clear" && draw != "tri" {
            return Err(map_backend_err(BackendError(format!(
                "raster draw must be `clear` or `tri`, got `{draw}`"
            ))));
        }
        if draw == "tri" && !self.graphics {
            return Err(map_backend_err(BackendError(
                "raster draw=tri requires a graphics queue".into(),
            )));
        }
        let r = f32_param(task, "r").unwrap_or(0.0);
        let g = f32_param(task, "g").unwrap_or(0.0);
        let b = f32_param(task, "b").unwrap_or(0.0);
        let a = f32_param(task, "a").unwrap_or(1.0);
        let need = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| map_backend_err(BackendError("raster size overflow".into())))?;
        let dst_bytes = self.buf_bytes(dst)?;
        if dst_bytes < need {
            return Err(map_backend_err(BackendError(format!(
                "raster destination {} has {dst_bytes} bytes, need at least {need}",
                dst.0
            ))));
        }

        if draw == "tri" {
            self.run_raster_tri(dst, width, height, need, [r, g, b, a])
        } else {
            self.run_raster_clear(dst, width, height, need, [r, g, b, a])
        }
    }

    fn run_raster_clear(
        &mut self,
        dst: ResourceId,
        width: u32,
        height: u32,
        need: usize,
        color: [f32; 4],
    ) -> Result<()> {
        unsafe {
            let (image, image_mem) = self.create_image(
                width,
                height,
                vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let (staging, staging_mem) = self.create_buffer(
                need,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            self.mark_sync_path();
            self.with_commands(|cmd| {
                let range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                let to_dst = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&to_dst),
                );
                let clear = vk::ClearColorValue { float32: color };
                self.device.cmd_clear_color_image(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &clear,
                    std::slice::from_ref(&range),
                );
                let to_src = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&to_src),
                );
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });
                self.device.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    staging,
                    std::slice::from_ref(&region),
                );
            })?;

            let pixels = self.map_staging_pixels(staging_mem, need)?;
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(staging_mem, None);
            self.device.destroy_image(image, None);
            self.device.free_memory(image_mem, None);
            self.upload(dst, &pixels)
        }
    }

    fn run_raster_tri(
        &mut self,
        dst: ResourceId,
        width: u32,
        height: u32,
        need: usize,
        clear_color: [f32; 4],
    ) -> Result<()> {
        let spirv = emit_raster_tri_spirv().map_err(map_emit_err)?;
        unsafe {
            let (image, image_mem) = self.create_image(
                width,
                height,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );
            let view = self
                .device
                .create_image_view(&view_info, None)
                .map_err(vk_err)?;

            let attachment = vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            let color_ref = vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let subpass = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(std::slice::from_ref(&color_ref));
            let dependency = vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
            let rp_info = vk::RenderPassCreateInfo::default()
                .attachments(std::slice::from_ref(&attachment))
                .subpasses(std::slice::from_ref(&subpass))
                .dependencies(std::slice::from_ref(&dependency));
            let render_pass = self
                .device
                .create_render_pass(&rp_info, None)
                .map_err(vk_err)?;

            let fb_attachments = [view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&fb_attachments)
                .width(width)
                .height(height)
                .layers(1);
            let framebuffer = self
                .device
                .create_framebuffer(&fb_info, None)
                .map_err(vk_err)?;

            let words: Vec<u32> = spirv
                .module
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
            let module = self
                .device
                .create_shader_module(&module_info, None)
                .map_err(vk_err)?;
            let vs_name = std::ffi::CString::new("vs_main").unwrap();
            let fs_name = std::ffi::CString::new("fs_main").unwrap();
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(module)
                    .name(&vs_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(module)
                    .name(&fs_name),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            };
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewports(std::slice::from_ref(&viewport))
                .scissors(std::slice::from_ref(&scissor));
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false);
            let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&color_blend_attachment));
            let layout_info = vk::PipelineLayoutCreateInfo::default();
            let pipeline_layout = self
                .device
                .create_pipeline_layout(&layout_info, None)
                .map_err(vk_err)?;
            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .color_blend_state(&color_blend)
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0);
            self.mark_pipeline_path();
            let pipelines = self
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_, e)| vk_err(e))?;
            let pipeline = pipelines[0];

            let (staging, staging_mem) = self.create_buffer(
                need,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            self.mark_sync_path();
            self.with_commands(|cmd| {
                let clear = vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: clear_color,
                    },
                };
                let begin_info = vk::RenderPassBeginInfo::default()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D { width, height },
                    })
                    .clear_values(std::slice::from_ref(&clear));
                self.device.cmd_begin_render_pass(
                    cmd,
                    &begin_info,
                    vk::SubpassContents::INLINE,
                );
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                self.device.cmd_draw(cmd, 3, 1, 0, 0);
                self.device.cmd_end_render_pass(cmd);

                let range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                let to_transfer = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&to_transfer),
                );

                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });
                self.device.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    staging,
                    std::slice::from_ref(&region),
                );
            })?;

            let pixels = self.map_staging_pixels(staging_mem, need)?;

            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_pipeline_layout(pipeline_layout, None);
            self.device.destroy_shader_module(module, None);
            self.device.destroy_framebuffer(framebuffer, None);
            self.device.destroy_render_pass(render_pass, None);
            self.device.destroy_image_view(view, None);
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(staging_mem, None);
            self.device.destroy_image(image, None);
            self.device.free_memory(image_mem, None);

            self.upload(dst, &pixels)
        }
    }

    unsafe fn map_staging_pixels(
        &self,
        staging_mem: vk::DeviceMemory,
        need: usize,
    ) -> Result<Vec<u8>> {
        let mut pixels = vec![0u8; need];
        let ptr = self
            .device
            .map_memory(staging_mem, 0, need as u64, vk::MemoryMapFlags::empty())
            .map_err(vk_err)?;
        std::ptr::copy_nonoverlapping(ptr.cast(), pixels.as_mut_ptr(), need);
        self.device.unmap_memory(staging_mem);
        Ok(pixels)
    }

    fn dispatch_compute(
        &mut self,
        spirv_bytes: &[u8],
        entry: &str,
        buffers: &[vk::Buffer],
        push_bytes: &[u8],
        groups_x: u32,
    ) -> Result<()> {
        unsafe {
            let words: Vec<u32> = spirv_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
            let module = self
                .device
                .create_shader_module(&module_info, None)
                .map_err(vk_err)?;

            let bindings: Vec<_> = (0..buffers.len() as u32)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            self.mark_descriptor_path();
            let set_layout = self
                .device
                .create_descriptor_set_layout(&set_layout_info, None)
                .map_err(vk_err)?;

            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_bytes.len() as u32);
            let mut layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&set_layout));
            if !push_bytes.is_empty() {
                layout_info = layout_info.push_constant_ranges(std::slice::from_ref(&push_range));
            }
            let pipeline_layout = self
                .device
                .create_pipeline_layout(&layout_info, None)
                .map_err(vk_err)?;

            let entry_c = std::ffi::CString::new(entry).map_err(|e| {
                map_backend_err(BackendError(format!("invalid entry `{entry}`: {e}")))
            })?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry_c.as_c_str());
            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout);
            self.mark_pipeline_path();
            let pipelines = self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_, e)| vk_err(e))?;
            let pipeline = pipelines[0];

            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(buffers.len() as u32)];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(1);
            let desc_pool = self
                .device
                .create_descriptor_pool(&pool_info, None)
                .map_err(vk_err)?;
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(std::slice::from_ref(&set_layout));
            let sets = self
                .device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(vk_err)?;
            let set = sets[0];

            let buffer_infos: Vec<_> = buffers
                .iter()
                .map(|b| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(*b)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                })
                .collect();
            let writes: Vec<_> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            self.device.update_descriptor_sets(&writes, &[]);

            self.with_commands(|cmd| {
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline_layout,
                    0,
                    std::slice::from_ref(&set),
                    &[],
                );
                if !push_bytes.is_empty() {
                    self.device.cmd_push_constants(
                        cmd,
                        pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push_bytes,
                    );
                }
                self.device.cmd_dispatch(cmd, groups_x, 1, 1);
            })?;

            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_pipeline_layout(pipeline_layout, None);
            self.device.destroy_descriptor_pool(desc_pool, None);
            self.device.destroy_descriptor_set_layout(set_layout, None);
            self.device.destroy_shader_module(module, None);
            Ok(())
        }
    }

    fn buf_bytes(&self, id: ResourceId) -> Result<usize> {
        self.buffers
            .get(&id)
            .map(|b| b.bytes)
            .ok_or_else(|| map_backend_err(BackendError(format!("resource {} not allocated", id.0))))
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let buffers = std::mem::take(&mut self.buffers);
            for (_, buf) in buffers {
                self.device.destroy_buffer(buf.buffer, None);
                self.device.free_memory(buf.memory, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
            .with(Feature::Bindless)
            .with(Feature::DescriptorBuffer)
            .with(Feature::DynamicRendering)
            .with(Feature::Synchronization2)
    }

    fn prepare(&mut self, graph: &Graph) -> Result<()> {
        for node in &graph.resources.nodes {
            let bytes = node.byte_size.ok_or_else(|| {
                map_backend_err(BackendError(format!(
                    "resource {} needs byte_size for vulkan backend",
                    node.id.0
                )))
            })? as usize;
            if let Some(existing) = self.buffers.get(&node.id) {
                if existing.bytes != bytes {
                    return Err(map_backend_err(BackendError(format!(
                        "resource {} size mismatch: have {}, need {bytes}",
                        node.id.0, existing.bytes
                    ))));
                }
                continue;
            }
            let (buffer, memory) = self.create_buffer(
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            self.buffers.insert(
                node.id,
                DeviceBuffer {
                    buffer,
                    memory,
                    bytes,
                },
            );
            let zeros = vec![0u8; bytes];
            self.upload(node.id, &zeros)?;
        }
        Ok(())
    }

    fn submit_task(&mut self, _graph: &Graph, task: &TaskNode) -> Result<()> {
        match &task.kind {
            TaskKind::Copy => self.run_copy(task),
            TaskKind::Fill => self.run_fill(task),
            TaskKind::MatMul => self.run_matmul(task),
            TaskKind::Raster => self.run_raster(task),
            TaskKind::RtTrace => Err(map_backend_err(BackendError(
                "ray tracing is not in the vulkan thin gate".into(),
            ))),
            other => Err(map_backend_err(BackendError(format!(
                "vulkan backend does not implement {other:?}"
            )))),
        }
    }
}

fn expect_bytes(id: ResourceId, have: usize, need: usize) -> Result<()> {
    if have != need {
        return Err(map_backend_err(BackendError(format!(
            "resource {} has {have} bytes, expected {need}",
            id.0
        ))));
    }
    Ok(())
}

fn vk_err(error: vk::Result) -> SchedulerError {
    SchedulerError::Backend("vulkan".into(), format!("{error:?}"))
}

fn map_emit_err(error: String) -> SchedulerError {
    SchedulerError::Backend("vulkan".into(), error)
}

fn map_backend_err(error: BackendError) -> SchedulerError {
    SchedulerError::Backend("vulkan".into(), error.0)
}
