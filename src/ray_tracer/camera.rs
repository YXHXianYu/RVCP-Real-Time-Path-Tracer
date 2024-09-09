use glam::Vec3;
use vulkano::{buffer::BufferContents, padded::Padded};

use super::prelude::*;

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
    pub position: [f32; 4],
    pub up: [f32; 4],
    pub look_at: [f32; 3],
    pub t_near: f32,
    pub t_far: f32,
    pub vertical_fov: f32,
    pub _padding: [u32; 2],
}

impl Camera {
    pub fn to_shader(&self) -> ray_tracer_shader::Camera {
        ray_tracer_shader::Camera {
            position: Padded::from(self.position.to_array()),
            up: Padded::from(self.up.to_array()),
            look_at: self.look_at.to_array(),
            t_near: self.t_near,
            t_far: self.t_far,
            vertical_fov: self.vertical_fov,
        }
    }
}