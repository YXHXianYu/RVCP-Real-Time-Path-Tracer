use vulkano::buffer::BufferContents;

use super::material::Material;

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: glam::Vec3,
    pub radius: f32,
    pub material: Material,
}

#[derive(BufferContents)]
#[repr(C)]
pub struct AlignedSphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub material: u32,
    pub _padding: [u32; 3],
}

impl Sphere {
    pub fn aligned(&self) -> AlignedSphere {
        // ray_tracer_shader::Sphere;
        AlignedSphere {
            center: self.center.to_array(),
            radius: self.radius,
            material: self.material.to_shader(),
            _padding: [0; 3],
        }
    }
}