use vulkan::Vk;

mod vulkan;
mod shader;

pub fn run() {
    println!("Running ray tracer");
    let vk = Vk::new();
    vk.run();
}