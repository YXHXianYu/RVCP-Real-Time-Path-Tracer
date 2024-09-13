mod ray_tracer;
mod vulkan;
mod shader;
mod config;
mod utils;

mod scene;

#[allow(unused_imports)]
mod prelude {
    pub use super::ray_tracer::*;
    pub use super::vulkan::*;
    pub use super::shader::*;
    pub use super::config::*;
    pub use super::utils::*;

    pub use super::scene::*;
}

pub use ray_tracer::run;
