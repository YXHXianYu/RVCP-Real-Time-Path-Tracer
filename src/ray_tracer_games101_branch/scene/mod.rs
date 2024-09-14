
mod camera;
mod material;
mod sphere;
mod mesh;

pub use camera::*;
pub use material::*;
pub use sphere::*;
pub use mesh::*;

use glam::{UVec3, Vec3, vec3};

pub struct Scene {
    pub camera: Camera,
    pub materials: Vec<Material>,
    pub spheres: Vec<Sphere>,
    pub mesh: Mesh,
}

impl Default for Scene {
    fn default() -> Self {
        let camera = Camera::new(
            Vec3::new(0.0, 274.0, -1050.0),
            Vec3::new(0.0, 274.0, 0.0),
            0.1,
            10000.0,
            40.0,

            100.0,
            5.0,
        );
        
        let materials = vec![
            Material::new_lambertian(Vec3::new(0.725, 0.71, 0.68)), // white
            Material::new_lambertian(Vec3::new(0.63, 0.065, 0.05)), // red
            Material::new_lambertian(Vec3::new(0.14, 0.45, 0.091)), // green
            Material::new_light(8.0 * Vec3::new(0.747 + 0.058, 0.747 + 0.258, 0.747) +
                                15.6 * Vec3::new(0.740 + 0.287, 0.740 + 0.160, 0.740) +
                                18.4 * Vec3::new(0.737 + 0.642, 0.737 + 0.159, 0.737)),
        ];

        let spheres: Vec<Sphere> = vec![];

        let cornel_height = 548.8;
        let cornel_width = 275.0;
        let cornel_light_width = 60.0;

        let tall_height = 330.0;
        let tall_v0 = vec3(423.0, 0.0, 247.0);
        let tall_v1 = vec3(265.0, 0.0, 296.0);
        let tall_v2 = vec3(314.0, 0.0, 456.0);
        let tall_v3 = vec3(472.0, 0.0, 406.0);
        let tall_v01_normal = (tall_v1 - tall_v0).cross(Vec3::Y).normalize();
        let tall_v12_normal = (tall_v2 - tall_v1).cross(Vec3::Y).normalize();
        let tall_v23_normal = (tall_v3 - tall_v2).cross(Vec3::Y).normalize();
        let tall_v30_normal = (tall_v0 - tall_v3).cross(Vec3::Y).normalize();

        let short_height = 165.0;
        let short_v0 = vec3(130.0, 0.0, 65.0);
        let short_v1 = vec3(82.0, 0.0, 225.0);
        let short_v2 = vec3(240.0, 0.0, 272.0);
        let short_v3 = vec3(290.0, 0.0, 114.0);
        let short_v01_normal = (short_v1 - short_v0).cross(Vec3::Y).normalize();
        let short_v12_normal = (short_v2 - short_v1).cross(Vec3::Y).normalize();
        let short_v23_normal = (short_v3 - short_v2).cross(Vec3::Y).normalize();
        let short_v30_normal = (short_v0 - short_v3).cross(Vec3::Y).normalize();

        let delta = vec3(-cornel_width, 0.0, -cornel_width);

        let mesh = Mesh{
            vertices: vec![
                // MARK: Cornel Box

                // top light
                Vertex { position: Vec3::new(-cornel_light_width, cornel_height - 0.01, -cornel_light_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(-cornel_light_width, cornel_height - 0.01, cornel_light_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(cornel_light_width, cornel_height - 0.01, cornel_light_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(cornel_light_width, cornel_height - 0.01, -cornel_light_width), normal: Vec3::new(0.0, -1.0, 0.0) },

                // top
                Vertex { position: Vec3::new(-cornel_width, cornel_height, -cornel_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(-cornel_width, cornel_height, cornel_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(cornel_width, cornel_height, cornel_width), normal: Vec3::new(0.0, -1.0, 0.0) },
                Vertex { position: Vec3::new(cornel_width, cornel_height, -cornel_width), normal: Vec3::new(0.0, -1.0, 0.0) },

                // left
                Vertex { position: Vec3::new(-cornel_width, 0.0, -cornel_width), normal: Vec3::new(1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(-cornel_width, 0.0, cornel_width), normal: Vec3::new(1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(-cornel_width, cornel_height, cornel_width), normal: Vec3::new(1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(-cornel_width, cornel_height, -cornel_width), normal: Vec3::new(1.0, 0.0, 0.0) },

                // right
                Vertex { position: Vec3::new(cornel_width, 0.0, -cornel_width), normal: Vec3::new(-1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(cornel_width, 0.0, cornel_width), normal: Vec3::new(-1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(cornel_width, cornel_height, cornel_width), normal: Vec3::new(-1.0, 0.0, 0.0) },
                Vertex { position: Vec3::new(cornel_width, cornel_height, -cornel_width), normal: Vec3::new(-1.0, 0.0, 0.0) },

                // front
                Vertex { position: Vec3::new(-cornel_width, 0.0, cornel_width), normal: Vec3::new(0.0, 0.0, -1.0) },
                Vertex { position: Vec3::new(cornel_width, 0.0, cornel_width), normal: Vec3::new(0.0, 0.0, -1.0) },
                Vertex { position: Vec3::new(cornel_width, cornel_height, cornel_width), normal: Vec3::new(0.0, 0.0, -1.0) },
                Vertex { position: Vec3::new(-cornel_width, cornel_height, cornel_width), normal: Vec3::new(0.0, 0.0, -1.0) },

                // back
                Vertex { position: Vec3::new(-cornel_width, 0.0, -cornel_width), normal: Vec3::new(0.0, 0.0, 1.0) },
                Vertex { position: Vec3::new(cornel_width, 0.0, -cornel_width), normal: Vec3::new(0.0, 0.0, 1.0) },
                Vertex { position: Vec3::new(cornel_width, cornel_height, -cornel_width), normal: Vec3::new(0.0, 0.0, 1.0) },
                Vertex { position: Vec3::new(-cornel_width, cornel_height, -cornel_width), normal: Vec3::new(0.0, 0.0, 1.0) },

                // bottom
                Vertex { position: Vec3::new(-cornel_width, 0.0, -cornel_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(-cornel_width, 0.0, cornel_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(cornel_width, 0.0, cornel_width), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: Vec3::new(cornel_width, 0.0, -cornel_width), normal: Vec3::new(0.0, 1.0, 0.0) },

                // MARK: Tall Box
                // top
                Vertex { position: delta + Vec3::new(tall_v0.x, tall_height, tall_v0.z), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: delta + Vec3::new(tall_v1.x, tall_height, tall_v1.z), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: delta + Vec3::new(tall_v2.x, tall_height, tall_v2.z), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: delta + Vec3::new(tall_v3.x, tall_height, tall_v3.z), normal: Vec3::new(0.0, 1.0, 0.0) },

                // v01
                Vertex { position: delta + tall_v0, normal: tall_v01_normal },
                Vertex { position: delta + tall_v1, normal: tall_v01_normal },
                Vertex { position: delta + tall_v1 + vec3(0.0, tall_height, 0.0), normal: tall_v01_normal },
                Vertex { position: delta + tall_v0 + vec3(0.0, tall_height, 0.0), normal: tall_v01_normal },

                // v12
                Vertex { position: delta + tall_v1, normal: tall_v12_normal },
                Vertex { position: delta + tall_v2, normal: tall_v12_normal },
                Vertex { position: delta + tall_v2 + vec3(0.0, tall_height, 0.0), normal: tall_v12_normal },
                Vertex { position: delta + tall_v1 + vec3(0.0, tall_height, 0.0), normal: tall_v12_normal },

                // v23
                Vertex { position: delta + tall_v2, normal: tall_v23_normal },
                Vertex { position: delta + tall_v3, normal: tall_v23_normal },
                Vertex { position: delta + tall_v3 + vec3(0.0, tall_height, 0.0), normal: tall_v23_normal },
                Vertex { position: delta + tall_v2 + vec3(0.0, tall_height, 0.0), normal: tall_v23_normal },

                // v30
                Vertex { position: delta + tall_v3, normal: tall_v30_normal },
                Vertex { position: delta + tall_v0, normal: tall_v30_normal },
                Vertex { position: delta + tall_v0 + vec3(0.0, tall_height, 0.0), normal: tall_v30_normal },
                Vertex { position: delta + tall_v3 + vec3(0.0, tall_height, 0.0), normal: tall_v30_normal },

                // MARK: Short Box
                // top
                Vertex { position: delta + Vec3::new(short_v0.x, short_height, short_v0.z), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: delta + Vec3::new(short_v1.x, short_height, short_v1.z), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: delta + Vec3::new(short_v2.x, short_height, short_v2.z), normal: Vec3::new(0.0, 1.0, 0.0) },
                Vertex { position: delta + Vec3::new(short_v3.x, short_height, short_v3.z), normal: Vec3::new(0.0, 1.0, 0.0) },

                // v01
                Vertex { position: delta + short_v0, normal: short_v01_normal },
                Vertex { position: delta + short_v1, normal: short_v01_normal },
                Vertex { position: delta + short_v1 + vec3(0.0, short_height, 0.0), normal: short_v01_normal },
                Vertex { position: delta + short_v0 + vec3(0.0, short_height, 0.0), normal: short_v01_normal },

                // v12
                Vertex { position: delta + short_v1, normal: short_v12_normal },
                Vertex { position: delta + short_v2, normal: short_v12_normal },
                Vertex { position: delta + short_v2 + vec3(0.0, short_height, 0.0), normal: short_v12_normal },
                Vertex { position: delta + short_v1 + vec3(0.0, short_height, 0.0), normal: short_v12_normal },

                // v23
                Vertex { position: delta + short_v2, normal: short_v23_normal },
                Vertex { position: delta + short_v3, normal: short_v23_normal },
                Vertex { position: delta + short_v3 + vec3(0.0, short_height, 0.0), normal: short_v23_normal },
                Vertex { position: delta + short_v2 + vec3(0.0, short_height, 0.0), normal: short_v23_normal },

                // v30
                Vertex { position: delta + short_v3, normal: short_v30_normal },
                Vertex { position: delta + short_v0, normal: short_v30_normal },
                Vertex { position: delta + short_v0 + vec3(0.0, short_height, 0.0), normal: short_v30_normal },
                Vertex { position: delta + short_v3 + vec3(0.0, short_height, 0.0), normal: short_v30_normal },
            ],
            faces: vec![
                // MARK: Cornel Box (Face)
                // top light
                Face { vertices: UVec3::new(0, 1, 2), material_id: 3 },
                Face { vertices: UVec3::new(0, 2, 3), material_id: 3 },

                // top
                Face { vertices: UVec3::new(4, 5, 6), material_id: 0 },
                Face { vertices: UVec3::new(4, 6, 7), material_id: 0 },

                // left
                Face { vertices: UVec3::new(8, 9, 10), material_id: 2 },
                Face { vertices: UVec3::new(8, 10, 11), material_id: 2 },

                // right
                Face { vertices: UVec3::new(12, 13, 14), material_id: 1 },
                Face { vertices: UVec3::new(12, 14, 15), material_id: 1 },

                // front
                Face { vertices: UVec3::new(16, 17, 18), material_id: 0 },
                Face { vertices: UVec3::new(16, 18, 19), material_id: 0 },

                // back
                // Face { vertices: UVec3::new(20, 21, 22), material_id: 0 },
                // Face { vertices: UVec3::new(20, 22, 23), material_id: 0 },

                // bottom
                Face { vertices: UVec3::new(24, 25, 26), material_id: 0 },
                Face { vertices: UVec3::new(24, 26, 27), material_id: 0 },

                // MARK: Tall Box (Face)
                // top
                Face { vertices: UVec3::new(28, 29, 30), material_id: 0 },
                Face { vertices: UVec3::new(28, 30, 31), material_id: 0 },

                // v01
                Face { vertices: UVec3::new(32, 33, 34), material_id: 0 },
                Face { vertices: UVec3::new(32, 34, 35), material_id: 0 },

                // v12
                Face { vertices: UVec3::new(36, 37, 38), material_id: 0 },
                Face { vertices: UVec3::new(36, 38, 39), material_id: 0 },

                // v23
                Face { vertices: UVec3::new(40, 41, 42), material_id: 0 },
                Face { vertices: UVec3::new(40, 42, 43), material_id: 0 },

                // v30
                Face { vertices: UVec3::new(44, 45, 46), material_id: 0 },
                Face { vertices: UVec3::new(44, 46, 47), material_id: 0 },

                // MARK: Short Box (Face)
                // top
                Face { vertices: UVec3::new(48, 49, 50), material_id: 0 },
                Face { vertices: UVec3::new(48, 50, 51), material_id: 0 },

                // v01
                Face { vertices: UVec3::new(52, 53, 54), material_id: 0 },
                Face { vertices: UVec3::new(52, 54, 55), material_id: 0 },

                // v12
                Face { vertices: UVec3::new(56, 57, 58), material_id: 0 },
                Face { vertices: UVec3::new(56, 58, 59), material_id: 0 },

                // v23
                Face { vertices: UVec3::new(60, 61, 62), material_id: 0 },
                Face { vertices: UVec3::new(60, 62, 63), material_id: 0 },

                // v30
                Face { vertices: UVec3::new(64, 65, 66), material_id: 0 },
                Face { vertices: UVec3::new(64, 66, 67), material_id: 0 },
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