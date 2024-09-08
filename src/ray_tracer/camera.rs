use glam::{Vec2, Vec3};
use vulkano::padded::Padded;
use super::shader::ray_tracer_shader;

pub struct Camera {
    pub position: Vec3,
    pub up: Vec3,
    pub look_at: Vec3,
    pub t_near: f32,
    pub t_far: f32,
    pub vertical_fov: f32,
    // pub time_start: f32, // what is this for?
    // pub time_end: f32,
    pub size: Vec2,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 3.0, 15.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            look_at: Vec3::new(0.0, 0.0, 0.0),
            t_near: 0.1,
            t_far: 10000.0,
            vertical_fov: 45.0,
            size: Vec2::new(800.0, 600.0),
        }
    }
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
            size: self.size.to_array(),
        }
    }
}