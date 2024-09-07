use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::Surface;
use vulkano::{Validated, VulkanLibrary};
use vulkano::instance::{Instance, InstanceCreateInfo};
use winit::event_loop::EventLoop;

pub fn create_device_etc() -> (
    Arc<Device>,
    Arc<StandardMemoryAllocator>,
    Arc<Queue>,
    StandardCommandBufferAllocator,
    EventLoop<()>,
    Arc<Instance>,
    Arc<PhysicalDevice>,
) {
    let event_loop = EventLoop::new();

    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Surface::required_extensions(&event_loop);

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    ).unwrap();

    let physical_device = instance
        .enumerate_physical_devices()
        .unwrap()
        .next()
        .unwrap();

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        })
        .unwrap() as u32;

    // println!("Found a graphics queue family at index {}.", queue_family_index);

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::empty()
            },
            ..Default::default()
        }
    ).unwrap();

    let memory_allocator = Arc::new(
        StandardMemoryAllocator::new_default(device.clone())
    );

    let queue = queues.next().unwrap();

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    (device, memory_allocator, queue, command_buffer_allocator, event_loop, instance, physical_device)
}

pub fn create_command_buffer_builder(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue_family_index: u32,
) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
    AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap()
}

pub fn create_compute_pipeline(
    device: Arc<Device>,
    shader_loader: fn(Arc<vulkano::device::Device>) -> Result<Arc<ShaderModule>, Validated<vulkano::VulkanError>>
) -> Arc<ComputePipeline> {
    let compute_shader = shader_loader(device.clone()).unwrap().entry_point("main").unwrap();
    let compute_pipeline_stage = PipelineShaderStageCreateInfo::new(compute_shader);
    let compute_pipeline_layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&compute_pipeline_stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    ).unwrap();
    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(compute_pipeline_stage, compute_pipeline_layout)
    ).unwrap()
}

pub fn save_image(image: ImageBuffer<Rgba<u8>, &[u8]>, path: &str) {
    let project_root = std::env::current_dir().unwrap();
    let image_path = project_root.join(path);
    image.save(image_path).unwrap();
}