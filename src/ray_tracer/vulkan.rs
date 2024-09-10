
use std::collections::HashMap;
use std::sync::Arc;
use glam::UVec3;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
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
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{self, PresentFuture, PresentMode, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError, VulkanLibrary};
use winit::dpi::PhysicalSize;
use winit::event::VirtualKeyCode;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use super::camera::AlignedCamera;
use super::scene::Scene;
use super::shader::*;

// === Runtime Info ===

type FencesType = Vec<Option<Arc<FenceSignalFuture<PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>>>>>;

pub struct RuntimeInfo {
    pub is_window_resized: bool,
    pub is_recreate_swapchain: bool,
    pub is_new_push_constants: bool,
    pub is_new_scene: bool,

    pub fences: FencesType,
    pub previous_idx: u32,

    pub fps_last_time: std::time::Instant,
    pub fps_frame_count: u32,

    pub last_tick_time: std::time::Instant,
    pub keyboard_is_pressing: HashMap<VirtualKeyCode, bool>,

    pub scene: Scene,
}

impl RuntimeInfo {
    fn new(images_len: u32, scene: Scene) -> Self {
        Self {
            is_window_resized: false,
            is_recreate_swapchain: false,
            is_new_push_constants: false,
            is_new_scene: false,

            fences: vec![None; images_len as usize],
            previous_idx: 0,

            fps_last_time: std::time::Instant::now(),
            fps_frame_count: 0,

            last_tick_time: std::time::Instant::now(),
            keyboard_is_pressing: HashMap::new(),

            scene,
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

    // buffers
    pub descriptor_set_0s: Vec<Arc<PersistentDescriptorSet>>,
}

#[derive(Debug, Clone, Copy, BufferContents)]
#[repr(C)]
struct PushConstant {
    camera: AlignedCamera,
    time: f32,
    // _padding: [u32; 3],
}

impl PushConstant {
    fn new(camera: AlignedCamera, time: f32) -> Self {
        Self {
            camera,
            time,
            // _padding: [0; 3],
        }
    }
}

impl Vk {
    pub fn new() -> (Vk, RuntimeInfo, EventLoop<()>) {
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
        println!("Max push constants size: {} Bytes", max_push_constants_size);

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

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_caps.min_image_count + 1,
                image_format,
                image_extent: window_size.into(),
                image_usage: ImageUsage::STORAGE, // Because of compute shader
                composite_alpha,
                present_mode: PresentMode::Immediate, // disable V-Sync
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
        
        // info
        let info = RuntimeInfo::new(
            images.len() as u32,
            Scene::default()
        );

        // buffers

        let descriptor_set_0s = Vk::create_descriptor_set_0s(
            &memory_allocator,
            &descriptor_set_allocator,
            &compute_pipeline,
            &images,
            &info,
        );

        let command_buffers = Vk::create_command_buffers(
            &descriptor_set_0s,
            &command_buffer_allocator,
            &queue,
            &compute_pipeline,
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

                descriptor_set_0s,
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

            info.is_new_scene = true; // Because images are changed
            // TODO: optimize
        }

        if info.is_new_scene {
            info.is_new_scene = false;

            let new_descriptor_set_0s = Vk::create_descriptor_set_0s(
                &self.memory_allocator,
                &self.descriptor_set_allocator,
                &self.compute_pipeline,
                &self.images,
                info,
            );
            self.descriptor_set_0s = new_descriptor_set_0s;
        }

        if info.is_window_resized || info.is_new_push_constants {
            info.is_window_resized = false;
            info.is_new_push_constants = true;
            
            let new_command_buffers = Vk::create_command_buffers(
                &self.descriptor_set_0s,
                &self.command_buffer_allocator,
                &self.queue,
                &self.compute_pipeline,
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
        descriptor_set_0s: &Vec<Arc<PersistentDescriptorSet>>,
        command_buffer_allocator: &StandardCommandBufferAllocator,
        queue: &Arc<Queue>,
        compute_pipeline: &Arc<ComputePipeline>,
        info: &RuntimeInfo,
        work_group: UVec3,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        descriptor_set_0s
            .iter()
            .map(|descriptor_set_0| {
                // push constants
                let push_constants = PushConstant::new(
                    info.scene.camera.aligned(),
                    (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64() % 1000.0) as f32,
                );
                // println!("Time: {}", push_constants.time);

                let mut builder = AutoCommandBufferBuilder::primary(
                    command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::MultipleSubmit,
                ).unwrap();

                builder
                    .bind_pipeline_compute(compute_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        compute_pipeline.layout().clone(),
                        0,
                        descriptor_set_0.clone(),
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

    fn create_descriptor_set_0s(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        compute_pipeline: &Arc<ComputePipeline>,
        images: &Vec<Arc<Image>>,
        info: &RuntimeInfo,
    ) -> Vec<Arc<PersistentDescriptorSet>> {
        images
            .iter()
            .map(|image| {
                let image_view = ImageView::new_default(image.clone()).unwrap();

                // buffers
                let length_buffer = Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    [info.scene.spheres.len() as u32, info.scene.point_lights.len() as u32],
                ).unwrap();
                let spheres_buffer = Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    info.scene.spheres.iter().map(|sphere| sphere.aligned()).collect::<Vec<_>>().into_iter()
                ).unwrap();
                let point_lights_buffer = Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    info.scene.point_lights.iter().map(|point_light| point_light.aligned()).collect::<Vec<_>>().into_iter()
                ).unwrap();
                let materials_buffer = Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    info.scene.materials.iter().map(|material| material.aligned()).collect::<Vec<_>>().into_iter()
                ).unwrap();
                
                // descriptor set
                let descriptor_set_0_layout 
                    = compute_pipeline.layout().set_layouts().get(0).unwrap();
                
                PersistentDescriptorSet::new(
                    descriptor_set_allocator,
                    descriptor_set_0_layout.clone(),
                    [
                        WriteDescriptorSet::image_view(0, image_view.clone()),
                        WriteDescriptorSet::buffer(1, length_buffer),
                        WriteDescriptorSet::buffer(2, spheres_buffer),
                        WriteDescriptorSet::buffer(3, point_lights_buffer),
                        WriteDescriptorSet::buffer(4, materials_buffer),
                    ],
                    [],
                ).unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn create_compute_pipeline(
        device: Arc<Device>,
        cs: Arc<ShaderModule>,
    ) -> Arc<ComputePipeline> {
        let compute_shader = cs.entry_point("main").unwrap();
        let compute_pipeline_stage = PipelineShaderStageCreateInfo::new(compute_shader.clone());
        let compute_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&compute_pipeline_stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        ).unwrap();

        // save to file
        // std::fs::write("compute_shader.txt", format!("{:#?}", compute_shader)).unwrap();
        // std::fs::write("compute_pipeline_stage.txt", format!("{:#?}", compute_pipeline_stage)).unwrap();
        // std::fs::write("compute_pipeline_layout.txt", format!("{:#?}", compute_pipeline_layout)).unwrap();
        // std::fs::write(
        //     "compute_pipeline_layout.set_layouts.txt",
        //     format!("{:#?}", compute_pipeline_layout.set_layouts())
        // ).unwrap();

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(compute_pipeline_stage, compute_pipeline_layout)
        ).unwrap()
    }
}