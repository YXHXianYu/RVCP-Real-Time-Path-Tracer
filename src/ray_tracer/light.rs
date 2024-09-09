use vulkano::padded::Padded;

use super::shader::ray_tracer_shader;

#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    pub position: glam::Vec3,
    pub energy: glam::Vec3,
}

impl PointLight {
    pub fn to_shader(&self) -> ray_tracer_shader::PointLight {
        ray_tracer_shader::PointLight {
            position: Padded::from(self.position.to_array()),
            energy: self.energy.to_array(),
        }
    }
}