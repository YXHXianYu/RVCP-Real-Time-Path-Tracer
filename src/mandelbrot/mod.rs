use config::Config;
use vulkan::Vk;

mod vulkan;
mod shader;
mod config;

#[allow(dead_code)]
pub fn run() {
    println!("Running Mandelbrot Set");
    let vk = Vk::new(Config::default());
    vk.run();
}