use vulkano::buffer::BufferContents;

use super::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    pub position: glam::Vec3,
    pub energy: glam::Vec3,
}

#[derive(Debug, Clone, Copy, BufferContents)]
#[repr(C)]
pub struct AlignedPointLight {
    pub position: [f32; 4],
    pub energy: [f32; 3],
    pub _padding: [u32; 1],
}


impl PointLight {
    pub fn aligned(&self) -> AlignedPointLight {
        AlignedPointLight {
            position: vec3_to_f32_4(self.position),
            energy: self.energy.to_array(),
            _padding: [0; 1],
        }
    }
}