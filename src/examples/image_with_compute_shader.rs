use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::CopyImageToBufferInfo;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};

use crate::examples::common::*;

mod cs_fractal {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
            #version 460
    
            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    
            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
    
            void main() {
                vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
                vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);
    
                vec2 z = vec2(0.0, 0.0);
                float i;
                for (i = 0.0; i < 1.0; i += 0.005) {
                    z = vec2(
                        z.x * z.x - z.y * z.y + c.x,
                        z.y * z.x + z.x * z.y + c.y
                    );
    
                    if (length(z) > 4.0) {
                        break;
                    }
                }
    
                vec4 to_write = vec4(vec3(i), 1.0);
                imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
            }
        "
    }
}

pub fn example_image_with_computer_shader(
    device: Arc<Device>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
) {
    const BINDING_POINT: u32 = 0;
    // const PICTURE_SIZE: [u32; 3] = [1024 * 20, 1024 * 20, 1];
    const PICTURE_SIZE: [u32; 3] = [1024, 1024, 1];
    const WORK_GROUP_SIZE: [u32; 3] = [PICTURE_SIZE[0]/8, PICTURE_SIZE[1]/8, PICTURE_SIZE[2]/1];
    
    // image (input / output)
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: PICTURE_SIZE,
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    ).unwrap();
    let image_view = ImageView::new_default(image.clone()).unwrap();

    println!("Image View Format: {:?}", image_view.format());

    // buffer (output)
    let buf = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..PICTURE_SIZE[0] * PICTURE_SIZE[1] * 4).map(|_| 0_u8),
    ).unwrap();

    // pipeline, descriptor set
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let compute_pipeline = create_compute_pipeline(
        device.clone(),
        cs_fractal::load
    );
    let set_layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        set_layout.clone(),
        [WriteDescriptorSet::image_view(BINDING_POINT, image_view.clone())],
        [],
    ).unwrap();

    // command
    let mut builder = create_command_buffer_builder(
        &command_buffer_allocator, queue.queue_family_index()
    );

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            BINDING_POINT,
            set
        )
        .unwrap()
        .dispatch(WORK_GROUP_SIZE)
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))
        .unwrap();
    
    let command_buffer = builder.build().unwrap();

    let start_time = std::time::Instant::now();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();

    let end_time = std::time::Instant::now();

    let buffer_content = buf.read().unwrap();
    let image_png = ImageBuffer::<Rgba<u8>, _>::from_raw(PICTURE_SIZE[0], PICTURE_SIZE[1], &buffer_content[..]).unwrap();
    save_image(image_png, "target/fractal.png");

    println!("Example: Image with computer shader: Success (Computing elapsed time: {:?})", end_time - start_time);
}