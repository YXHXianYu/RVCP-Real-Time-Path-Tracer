
mod copy_buffer;
mod compute_shader;
mod image;
mod image_with_compute_shader;
mod graphics_pipeline;
mod window;
mod common;

use copy_buffer::*;
use compute_shader::*;
use image::*;
use image_with_compute_shader::*;
use graphics_pipeline::*;
use window::*;

use common::*;

#[allow(dead_code)]
pub fn examples() {
    let (
        device,
        memory_allocator,
        queue,
        command_buffer_allocator,
        event_loop,
        instance,
        physical_device,
    ) = create_device_etc();

    example_copy_buffer(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
    );

    example_compute_shader(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
    );

    example_image(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
    );

    example_image_with_computer_shader(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
    );

    example_graphics_pipeline(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
    );

    // all borrowed here
    example_window(
        device,
        memory_allocator,
        queue,
        command_buffer_allocator,
        event_loop,
        instance,
        physical_device,
    );
}