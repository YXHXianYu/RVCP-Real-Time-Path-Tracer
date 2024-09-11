use glam::Vec3;
use vulkano::buffer::BufferContents;

use crate::ray_tracer::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: Vec3,
    pub up: Vec3,
    pub look_at: Vec3,
    pub t_near: f32,
    pub t_far: f32,
    pub vertical_fov: f32,
    // pub time_start: f32, // what is this for?
    // pub time_end: f32,
    // pub size: Vec2,
}

#[derive(Debug, Clone, Copy, BufferContents)]
#[repr(C)]
pub struct AlignedCamera {
    position: [f32; 4],
    up: [f32; 4],
    look_at: [f32; 3],
    t_near: f32,
    t_far: f32,
    vertical_fov: f32,
    _padding: [u32; 2],
}

impl Camera {
    pub fn aligned(&self) -> AlignedCamera {
        AlignedCamera {
            position: vec3_to_f32_4(self.position),
            up: vec3_to_f32_4(self.up),
            look_at: self.look_at.to_array(),
            t_near: self.t_near,
            t_far: self.t_far,
            vertical_fov: self.vertical_fov,
            _padding: [0; 2],
        }
    }
}