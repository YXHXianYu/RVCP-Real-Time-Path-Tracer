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

    pub move_speed: f32,
    pub rotate_speed: f32,

    // derived
    pub forward: Vec3,
    pub right: Vec3,
    pub yaw: f32,
    pub pitch: f32,
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

    pub fn new(
        position: Vec3,
        up: Vec3,
        look_at: Vec3,
        t_near: f32,
        t_far: f32,
        vertical_fov: f32,

        move_speed: f32,
        rotate_speed: f32,
    ) -> Self {
        let right = (look_at - position).cross(up).normalize();
        let forward = up.cross(right).normalize();
        // be careful of 0.atan()
        let yaw = forward.z.atan2(forward.x).to_degrees();
        let pitch = forward.y.asin().to_degrees();

        println!("right: {:?}", right);
        println!("forward: {:?}", forward);
        println!("yaw: {:?}", yaw);
        println!("pitch: {:?}", pitch);

        Self {
            position,
            up,
            look_at,
            t_near,
            t_far,
            vertical_fov,

            move_speed,
            rotate_speed,

            forward,
            right,
            yaw,
            pitch,
        }
    }
}