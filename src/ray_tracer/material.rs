use glam::Vec3;
use vulkano::buffer::BufferContents;

#[derive(Debug, Clone, Copy)]
pub enum MaterialType {
    Lambertian = 0,
    Metal,
    Dielectric,
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    ty: MaterialType,
    albedo: Vec3,
    fuzz: f32,
    index_of_refraction: f32,
}

#[derive(Debug, Clone, Copy, BufferContents)]
#[repr(C)]
pub struct AlignedMaterial {
    albedo: [f32; 3],
    ty: u32,
    fuzz: f32,
    index_of_refraction: f32,
    _padding: [u32; 2],
}

impl Material {
    pub fn aligned(&self) -> AlignedMaterial {
        // ray_tracer_shader::Material
        AlignedMaterial {
            albedo: self.albedo.to_array(),
            ty: self.ty as u32,
            fuzz: self.fuzz,
            index_of_refraction: self.index_of_refraction,
            _padding: [0; 2],
        }
    }

    pub fn new_lambertian(albedo: Vec3) -> Self {
        Self {
            albedo,
            ty: MaterialType::Lambertian,
            fuzz: 0.0,
            index_of_refraction: 0.0,
        }
    }

    pub fn new_metal(albedo: Vec3, fuzz: f32) -> Self {
        Self {
            albedo,
            ty: MaterialType::Metal,
            fuzz,
            index_of_refraction: 0.0,
        }
    }

    pub fn new_dielectric(index_of_refraction: f32) -> Self {
        Self {
            albedo: Vec3::new(1.0, 1.0, 1.0),
            ty: MaterialType::Dielectric,
            fuzz: 0.0,
            index_of_refraction,
        }
    }
}