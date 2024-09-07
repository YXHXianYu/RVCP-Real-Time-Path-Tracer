use config::Config;
use vulkan::Vk;

mod vulkan;
mod shader;
mod config;

pub fn run() {
    println!("Running ray tracer");
    let vk = Vk::new(Config::default());
    vk.run();
}