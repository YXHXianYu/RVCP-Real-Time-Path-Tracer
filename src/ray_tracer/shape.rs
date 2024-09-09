use super::{material::Material, shader::ray_tracer_shader};

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: glam::Vec3,
    pub radius: f32,
    pub material: Material,
}

impl Sphere {
    pub fn to_shader(&self) -> ray_tracer_shader::Sphere {
        ray_tracer_shader::Sphere {
            center: self.center.to_array(),
            radius: self.radius,
            material: self.material.to_shader(),
        }
    }
}