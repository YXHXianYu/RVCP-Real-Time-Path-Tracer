use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::CopyBufferInfo;
use vulkano::device::{Device, Queue};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::{self, GpuFuture};

use crate::examples::common::*;

pub fn example_copy_buffer(
    device: Arc<Device>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
) {
    // Buffer
    let source_content: Vec<i32> = (0..64).collect();
    let source = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        source_content,
    ).unwrap();

    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination = Buffer::from_iter(
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
        destination_content,
    ).unwrap();

    // Command Buffer
    let mut builder = create_command_buffer_builder(
        &command_buffer_allocator,
        queue.queue_family_index()
    );
    builder
        .copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))
        .unwrap();
    let command_buffer = builder.build().unwrap();

    // Submission
    let fence_signal_future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush() // Here, the command buffer is actually executed.
        .unwrap();

    // Synchonization (Wait until the copy is finished)
    fence_signal_future.wait(None).unwrap();

    let src_content = source.read().unwrap();
    let dst_content = destination.read().unwrap();
    assert_eq!(&*src_content, &*dst_content);
    println!("Example: Copy buffer: Success");
}