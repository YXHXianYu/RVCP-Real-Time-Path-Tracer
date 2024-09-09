mod ray_tracer;
mod vulkan;
mod shader;
mod config;
mod utils;

mod scene;
mod camera;
mod shape;
mod light;
mod material;

#[allow(unused_imports)]
mod prelude {
    pub use super::ray_tracer::*;
    pub use super::vulkan::*;
    pub use super::shader::*;
    pub use super::config::*;
    pub use super::utils::*;

    pub use super::scene::*;
    pub use super::camera::*;
    pub use super::shape::*;
    pub use super::light::*;
    pub use super::material::*;
}

pub use ray_tracer::run;
