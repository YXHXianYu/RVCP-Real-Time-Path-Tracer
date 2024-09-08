
use std::collections::HashMap;
use std::sync::Arc;
use glam::UVec3;
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{self, PresentFuture, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError, VulkanLibrary};
use winit::dpi::PhysicalSize;
use winit::event::VirtualKeyCode;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use super::camera::Camera;
use super::config::Config;
use super::shader::*;

// === Runtime Info ===

type FencesType = Vec<Option<Arc<FenceSignalFuture<PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>>>>>;

pub struct RuntimeInfo {
    pub is_window_resized: bool,
    pub is_recreate_swapchain: bool,
    pub is_new_push_constants: bool,

    pub fences: FencesType,
    pub previous_idx: u32,

    pub fps_last_time: std::time::Instant,
    pub fps_frame_count: u32,

    pub last_tick_time: std::time::Instant,
    pub keyboard_is_pressing: HashMap<VirtualKeyCode, bool>,

    pub camera: Camera,
}

impl RuntimeInfo {
    fn new(images_len: u32) -> Self {
        Self {
            is_window_resized: false,
            is_recreate_swapchain: false,
            is_new_push_constants: false,

            fences: vec![None; images_len as usize],
            previous_idx: 0,

            fps_last_time: std::time::Instant::now(),
            fps_frame_count: 0,

            last_tick_time: std::time::Instant::now(),
            keyboard_is_pressing: HashMap::new(),

            camera: Default::default(),
        }
    }
}

// === Vulkan ===

#[allow(dead_code)]
pub struct Vk {
    // basic
    pub instance: Arc<Instance>,
    pub physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue_family_index: u32,
    pub queue: Arc<Queue>,

    // window
    pub window: Arc<Window>,
    pub surface: Arc<Surface>,
    pub swapchain: Arc<Swapchain>,
    pub window_size: PhysicalSize<u32>,

    // allocator
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: StandardCommandBufferAllocator,
    pub descriptor_set_allocator: StandardDescriptorSetAllocator,

    // render
    pub compute_pipeline: Arc<ComputePipeline>,
    pub images: Vec<Arc<Image>>,
    pub image_format: Format,
    pub command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,

    // config
    pub config: Config,
}

impl Vk {
    pub fn new(config: Config) -> (Vk, RuntimeInfo, EventLoop<()>) {
        let event_loop = EventLoop::new();

        // basic
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

        let max_push_constants_size = physical_device
            .properties()
            .max_push_constants_size;
        println!("Max push constants size: {}", max_push_constants_size);

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
            })
            .unwrap() as u32;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: DeviceExtensions {
                    khr_swapchain: true,
                    khr_swapchain_mutable_format: true,
                    ..DeviceExtensions::empty()
                },
                ..Default::default()
            }
        ).unwrap();

        let queue = queues.next().unwrap();

        // window
        let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        
        let window_size = window.inner_size();
        let surface_caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        let composite_alpha = surface_caps.supported_composite_alpha.into_iter().next().unwrap();

        // physical_device.surface_formats(&surface, Default::default()).unwrap().iter().for_each(|(format, color_space)| {
        //     println!("Format: {:?}, Color Space: {:?}", format, color_space);
        // });
        // println!("Image Format: {:?}", image_format);

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_caps.min_image_count + 1,
                image_format,
                image_extent: window_size.into(),
                image_usage: ImageUsage::STORAGE, // Because of compute shader
                composite_alpha,
                ..Default::default()
            },
        ).unwrap();

        // allocator
        let memory_allocator = Arc::new(
            StandardMemoryAllocator::new_default(device.clone())
        );

        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );
        
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default()
        );

        // render
        let compute_pipeline = Vk::create_compute_pipeline(
            device.clone(),
            ray_tracer_shader::load(device.clone()).unwrap()
        );

        
        let info = RuntimeInfo::new(images.len() as u32);

        let command_buffers = Vk::create_command_buffers(
            &descriptor_set_allocator,
            &command_buffer_allocator,
            &queue,
            &compute_pipeline,
            &images,
            &info,
            UVec3::new(window_size.width / 8, window_size.height / 8, 1),
        );

        (
            Vk {
                instance,
                physical_device,
                device,
                queue_family_index,
                queue,

                window,
                surface,
                swapchain,
                window_size,

                memory_allocator,
                command_buffer_allocator,
                descriptor_set_allocator,

                compute_pipeline,
                images,
                image_format,
                command_buffers,

                config,
            },
            info,
            event_loop,
        )
    }

    pub fn update_frame(&mut self, info: &mut RuntimeInfo) {
        if info.is_window_resized || info.is_recreate_swapchain {
            let new_window_size = self.window.inner_size();
            if new_window_size.width == 0 || new_window_size.height == 0 {
                return;
            }

            self.window_size = new_window_size;

            info.is_recreate_swapchain = false;

            let (new_swapchain, new_images) = self.swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: self.window_size.into(),
                    ..self.swapchain.create_info()
                })
                .unwrap();
            self.swapchain = new_swapchain;
            self.images = new_images;
        }

        if info.is_window_resized || info.is_new_push_constants {
            info.is_window_resized = false;
            info.is_new_push_constants = false;
            
            let new_command_buffers = Vk::create_command_buffers(
                &self.descriptor_set_allocator,
                &self.command_buffer_allocator,
                &self.queue,
                &self.compute_pipeline,
                &self.images,
                info,
                UVec3::new(self.window_size.width / 8, self.window_size.height / 8, 1),
            );
            self.command_buffers = new_command_buffers;
        }

        let (current_idx, suboptimal, acquire_future)
            = match swapchain::acquire_next_image(self.swapchain.clone(), None)
                .map_err(Validated::unwrap)
        {
            Ok(r) => r,
            Err(VulkanError::OutOfDate) => {
                info.is_recreate_swapchain = true;
                return;
            },
            Err(e) => panic!("{:?}", e),
        };
        
        if suboptimal {
            info.is_recreate_swapchain = true;
        }

        // ?
        if let Some(image_fence) = &info.fences[current_idx as usize] {
            image_fence.wait(None).unwrap();
        }

        let previous_future = match info.fences[info.previous_idx as usize].clone() {
            None => {
                let mut now = sync::now(self.device.clone());
                now.cleanup_finished();

                now.boxed()
            },
            Some(fence) => fence.boxed(),
        };

        let future = previous_future
            .join(acquire_future)
            .then_execute(self.queue.clone(), self.command_buffers[current_idx as usize].clone())
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    current_idx
                ),
            )
            .then_signal_fence_and_flush();
        
            info.fences[current_idx as usize] = match future.map_err(Validated::unwrap) {
            Ok(value) => Some(Arc::new(value)),
            Err(VulkanError::OutOfDate) => {
                info.is_recreate_swapchain = true;
                None
            },
            Err(e) => panic!("{:?}", e),
        };

        info.previous_idx = current_idx;
    }

    fn create_command_buffers(
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        command_buffer_allocator: &StandardCommandBufferAllocator,
        queue: &Arc<Queue>,
        compute_pipeline: &Arc<ComputePipeline>,
        images: &Vec<Arc<Image>>,
        info: &RuntimeInfo,
        work_group: UVec3,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        images
            .iter()
            .map(|image| {
                let image_view = ImageView::new_default(image.clone()).unwrap();

                // println!("Image view format: {:?}", image_view.format());

                let descriptor_set_layout 
                    = compute_pipeline.layout().set_layouts().get(0).unwrap();
                
                let descriptor_set = PersistentDescriptorSet::new(
                    descriptor_set_allocator,
                    descriptor_set_layout.clone(),
                    [WriteDescriptorSet::image_view(0, image_view.clone())],
                    [],
                ).unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::MultipleSubmit,
                ).unwrap();

                let push_constants = info.camera.to_shader();

                builder
                    .bind_pipeline_compute(compute_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        compute_pipeline.layout().clone(),
                        0,
                        descriptor_set,
                    )
                    .unwrap()
                    .push_constants(
                        compute_pipeline.layout().clone(),
                        0,
                        push_constants,
                    )
                    .unwrap()
                    .dispatch(work_group.to_array())
                    .unwrap();

                builder.build().unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn create_compute_pipeline(
        device: Arc<Device>,
        cs: Arc<ShaderModule>,
    ) -> Arc<ComputePipeline> {
        let compute_shader = cs.entry_point("main").unwrap();
        let compute_pipeline_stage = PipelineShaderStageCreateInfo::new(compute_shader.clone());

        // save compute_pipeline_stage to a txt file
        // std::fs::write("compute_pipeline_stage.txt", format!("{:#?}", compute_pipeline_stage)).unwrap();
        // std::fs::write("compute_shader_entry_point.txt", format!("{:#?}", compute_shader)).unwrap();

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
}