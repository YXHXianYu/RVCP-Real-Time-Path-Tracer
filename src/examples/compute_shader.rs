use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};

use crate::examples::common::*;

mod cs_multiple {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
            #version 460
    
            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
    
            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;
            
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                for (int i = 1; i <= 100; i++) {
                    buf.data[idx] += i;
                }
            }
        ",
    }
}

pub fn example_compute_shader(
    device: Arc<Device>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
) {
    const WORK_GROUP_SIZE: u32 = 64; // must match the local_size_x in the shader manually
    // const DATA_SIZE: u32 = 1_u32 << 28; // 3e8
    const DATA_SIZE: u32 = 1_u32 << 20; // 1e6
    const WORK_GROUP_COUNT: u32 = DATA_SIZE / WORK_GROUP_SIZE;

    // println!("Data size: {}", DATA_SIZE);

    // data
    let data_iter = 0..DATA_SIZE;
    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_iter
    ).unwrap();

    // shader & pipeline
    let compute_pipeline = create_compute_pipeline(device.clone(), cs_multiple::load);
    let compute_pipeline_layout = compute_pipeline.layout();

    // descriptor set
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let descriptor_set_layouts = compute_pipeline_layout.set_layouts();
    let descriptor_set_layout_index = 0; // set 0
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // binding 0
        [],
    ).unwrap();

    // dispatch
    let work_group_count = [WORK_GROUP_COUNT, 1, 1];

    let mut builder = create_command_buffer_builder(
        &command_buffer_allocator,
        queue.queue_family_index()
    );
    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline_layout.clone(),
            descriptor_set_layout_index as u32,
            descriptor_set
        )
        .unwrap()
        .dispatch(work_group_count)
        .unwrap();

    let command_buffer = builder.build().unwrap();

    // start time
    let start_time = std::time::Instant::now();

    let fence_signal_future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    fence_signal_future.wait(None).unwrap();

    let end_time = std::time::Instant::now();

    let _data = data_buffer.read().unwrap();
    // for i in 0..5 {
    //     println!("Data[{}] = {}", i, data[i]);
    // }
    // for i in (DATA_SIZE - 5)..DATA_SIZE {
    //     println!("Data[{}] = {}", i, data[i as usize]);
    // }

    println!("Example: Compute shader: Success (Elapsed time: {:?})", end_time - start_time);
}