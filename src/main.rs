
mod examples;
mod common;

use common::*;

fn main() {

    let (
        device,
        memory_allocator,
        queue,
        command_buffer_allocator,
        event_loop,
        instance,
    ) = create_device_etc();

    examples::example_copy_buffer(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
    );

    examples::example_compute_shader(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
    );

    examples::example_image(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
    );

    examples::example_image_with_computer_shader(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
    );

    examples::example_graphics_pipeline(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
    );

    examples::example_window(
        device.clone(),
        memory_allocator.clone(),
        queue.clone(),
        &command_buffer_allocator,
        event_loop,
        instance.clone(),
    )
}
