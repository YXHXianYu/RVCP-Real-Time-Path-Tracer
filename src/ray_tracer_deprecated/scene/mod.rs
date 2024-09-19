
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
            Vec3::new(0.0, 0.0, 0.0),
            0.1,
            1000.0,
            120.0,

            3.0,
            10.0,
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
            // Material::new_light(8.0 * Vec3::new(0.747 + 0.058, 0.747 + 0.258, 0.747) +
            //                     15.6 * Vec3::new(0.740 + 0.287, 0.740 + 0.160, 0.740) +
            //                     18.4 * Vec3::new(0.737 + 0.642, 0.737 + 0.159, 0.737)),
            Material::new_light(Vec3::new(1.0, 1.0, 1.0)),
            Material::new_lambertian(Vec3::new(1.0, 0.0, 0.0)), // red
            Material::new_lambertian(Vec3::new(0.0, 1.0, 0.0)), // green
        ];

        let spheres: Vec<Sphere> = vec![
            // Sphere {
            //     center: Vec3::new(0.0, -1000.0, 0.0),
            //     radius: 1000.0,
            //     material_id: 0,
            // },
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

        let roof_height = 5.0;
        let roof_width = 5.0;
        let roof_light_width = 5.0;
        // let roof_width = 0.01;
        // let roof_light_width = 0.01;
        let mesh = Mesh{
            vertices: vec![
                // top light
                Vertex { position: Vec3::new(-roof_light_width, roof_height - 0.01, -roof_light_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(-roof_light_width, roof_height - 0.01, roof_light_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(roof_light_width, roof_height - 0.01, roof_light_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(roof_light_width, roof_height - 0.01, -roof_light_width), normal: Vec3::new(0.0, 1.0, 0.0) },

                // top
                Vertex { position: Vec3::new(-roof_width, roof_height, -roof_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(-roof_width, roof_height, roof_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(roof_width, roof_height, roof_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(roof_width, roof_height, -roof_width), normal: Vec3::new(0.0, -1.0, 0.0) },

                // left
                Vertex { position: Vec3::new(-roof_width, 0.0, -roof_width), normal: Vec3::new(1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(-roof_width, 0.0, roof_width), normal: Vec3::new(1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(-roof_width, roof_height, roof_width), normal: Vec3::new(1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(-roof_width, roof_height, -roof_width), normal: Vec3::new(1.0, 0.0, 0.0) },

                // right
                Vertex { position: Vec3::new(roof_width, 0.0, -roof_width), normal: Vec3::new(-1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(roof_width, 0.0, roof_width), normal: Vec3::new(-1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(roof_width, roof_height, roof_width), normal: Vec3::new(-1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(roof_width, roof_height, -roof_width), normal: Vec3::new(-1.0, 0.0, 0.0) },

                // front
                Vertex { position: Vec3::new(-roof_width, 0.0, roof_width), normal: Vec3::new(0.0, 0.0, -1.0) },
                Vertex { position: Vec3::new(roof_width, 0.0, roof_width), normal: Vec3::new(0.0, 0.0, -1.0) },
                Vertex { position: Vec3::new(roof_width, roof_height, roof_width), normal: Vec3::new(0.0, 0.0, -1.0) },
                Vertex { position: Vec3::new(-roof_width, roof_height, roof_width), normal: Vec3::new(0.0, 0.0, -1.0) },

                // back
                Vertex { position: Vec3::new(-roof_width, 0.0, -roof_width), normal: Vec3::new(0.0, 0.0, 1.0) },
                Vertex { position: Vec3::new(roof_width, 0.0, -roof_width), normal: Vec3::new(0.0, 0.0, 1.0) },
                Vertex { position: Vec3::new(roof_width, roof_height, -roof_width), normal: Vec3::new(0.0, 0.0, 1.0) },
                Vertex { position: Vec3::new(-roof_width, roof_height, -roof_width), normal: Vec3::new(0.0, 0.0, 1.0) },

                // bottom
                Vertex { position: Vec3::new(-roof_width, 0.0, -roof_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(-roof_width, 0.0, roof_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(roof_width, 0.0, roof_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(roof_width, 0.0, -roof_width), normal: Vec3::new(0.0, 1.0, 0.0) },
            ],
            faces: vec![
                // top light
                Face { vertices: UVec3::new(0, 1, 2), material_id: 8 },
                Face { vertices: UVec3::new(0, 2, 3), material_id: 8 },

                // top
                Face { vertices: UVec3::new(4, 5, 6), material_id: 0 },
                Face { vertices: UVec3::new(4, 6, 7), material_id: 0 },

                // left
                Face { vertices: UVec3::new(8, 9, 10), material_id: 9 },
                Face { vertices: UVec3::new(8, 10, 11), material_id: 9 },

                // right
                Face { vertices: UVec3::new(12, 13, 14), material_id: 10 },
                Face { vertices: UVec3::new(12, 14, 15), material_id: 10 },

                // front
                // Face { vertices: UVec3::new(16, 17, 18), material_id: 0 },
                // Face { vertices: UVec3::new(16, 18, 19), material_id: 0 },

                // back
                Face { vertices: UVec3::new(20, 21, 22), material_id: 0 },
                Face { vertices: UVec3::new(20, 22, 23), material_id: 0 },

                // bottom
                Face { vertices: UVec3::new(24, 25, 26), material_id: 0 },
                Face { vertices: UVec3::new(24, 26, 27), material_id: 0 },
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