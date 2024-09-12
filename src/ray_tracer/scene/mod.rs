
mod camera;
mod material;
mod sphere;
mod mesh;

pub use camera::*;
pub use material::*;
pub use sphere::*;
pub use mesh::*;

use glam::{UVec3, Vec3};

pub struct Scene {
    pub camera: Camera,
    pub materials: Vec<Material>,
    pub spheres: Vec<Sphere>,
    pub mesh: Mesh,
}

impl Default for Scene {
    fn default() -> Self {
        let camera = Camera::new(
            Vec3::new(0.0, 1.0, 3.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            0.1,
            10000.0,
            90.0,

            0.1,
            0.1,
        );
        
        let materials = vec![
            Material::new_lambertian(Vec3::new(1.0, 1.0, 1.0)),
            Material::new_lambertian(Vec3::new(0.8, 0.3, 0.3)),
            Material::new_lambertian(Vec3::new(0.3, 0.7, 0.3)),
            Material::new_metal(Vec3::new(0.8, 0.8, 0.8), 0.3),
            Material::new_metal(Vec3::new(1.0, 1.0, 1.0), 0.0),
            Material::new_metal(Vec3::new(0.5, 0.4, 0.9), 0.3),
            Material::new_dielectric(1.3),
            Material::new_dielectric(2.5),
            Material::new_light(Vec3::new(1.0, 1.0, 1.0)),
        ];

        let spheres: Vec<Sphere> = vec![
            Sphere {
                center: Vec3::new(0.0, -1000.0, 0.0),
                radius: 1000.0,
                material_id: 0,
            },
            Sphere {
                center: Vec3::new(0.0, 1.0, 0.0),
                radius: 1.0,
                material_id: 1,
            },
            Sphere {
                center: Vec3::new(-1.5, 0.5, 2.0),
                radius: 0.5,
                material_id: 2,
            },
            Sphere {
                center: Vec3::new(-2.0, 1.0, 0.0),
                radius: 1.0,
                material_id: 3,
            },
            Sphere {
                center: Vec3::new(0.0, 0.25, 1.75),
                radius: 0.25,
                material_id: 4,
            },
            Sphere {
                center: Vec3::new(1.5, 0.25, 1.75),
                radius: 0.25,
                material_id: 5,
            },
            Sphere {
                center: Vec3::new(1.25, 0.25, 1.25),
                radius: 0.25,
                material_id: 6,
            },
            Sphere {
                center: Vec3::new(2.0, 1.0, 0.0),
                radius: 1.0,
                material_id: 7,
            },
            Sphere {
                center: Vec3::new(-1.0, 0.25, 1.0),
                radius: 0.25,
                material_id: 8,
            },
        ];

        let roof_height = 3.0;
        let roof_width = 5.0;
        let roof_light_width = 0.5;
        let mesh = Mesh{
            vertices: vec![
                Vertex { position: Vec3::new(-roof_width, roof_height, -roof_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(-roof_width, roof_height, roof_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(roof_width, roof_height, roof_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(roof_width, roof_height, -roof_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                
                Vertex { position: Vec3::new(-roof_light_width, roof_height - 0.01, -roof_light_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(-roof_light_width, roof_height - 0.01, roof_light_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(roof_light_width, roof_height - 0.01, roof_light_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(roof_light_width, roof_height - 0.01, -roof_light_width), normal: Vec3::new(0.0, 1.0, 0.0) },
            ],
            faces: vec![
                Face { vertices: UVec3::new(0, 1, 2), material_id: 0 },
                Face { vertices: UVec3::new(0, 2, 3), material_id: 0 },

                Face { vertices: UVec3::new(4, 5, 6), material_id: 8 },
                Face { vertices: UVec3::new(4, 6, 7), material_id: 8 },
            ],
        };

        Self {
            camera,
            materials,
            spheres,
            mesh,
        }
    }
}