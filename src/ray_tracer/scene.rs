use glam::Vec3;

use super::{camera::Camera, light::PointLight, material::Material, shape::Sphere};

pub struct Scene {
    pub camera: Camera,
    pub spheres: Vec<Sphere>,
    pub point_lights: Vec<PointLight>,
}

impl Default for Scene {
    fn default() -> Self {
        let camera = Camera {
            position: Vec3::new(0.0, 2.0, 5.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            look_at: Vec3::new(0.0, 0.0, 0.0),
            t_near: 0.1,
            t_far: 10000.0,
            vertical_fov: 90.0,
        };

        let spheres: Vec<Sphere> = vec![
            Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 1.0,
                material: Material::matte(),
            },
            Sphere {
                center: Vec3::new(2.0, 0.0, 0.0),
                radius: 1.0,
                material: Material::matte(),
            },
        ];

        let point_lights: Vec<PointLight> = vec![
            PointLight {
                position: Vec3::new(0.0, 5.0, 0.0),
                energy: Vec3::new(1.0, 1.0, 1.0),
            },
            PointLight {
                position: Vec3::new(0.0, 3.0, 5.0),
                energy: Vec3::new(1.0, 1.0, 1.0),
            },
        ];

        Self {
            camera,
            spheres,
            point_lights
        }
    }
}