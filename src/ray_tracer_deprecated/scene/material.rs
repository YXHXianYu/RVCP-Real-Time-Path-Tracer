use glam::Vec3;
use vulkano::buffer::BufferContents;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaterialType {
    Lambertian = 0,
    Metal,
    Dielectric,
    Light,
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub ty: MaterialType,
    pub albedo: Vec3,
    pub fuzz: f32,
    pub refraction_ratio: f32,
}

#[derive(Debug, Clone, Copy, BufferContents)]
#[repr(C)]
pub struct AlignedMaterial {
    albedo: [f32; 3],
    ty: u32,
    fuzz: f32,
    refraction_ratio: f32,
    _padding: [u32; 2],
}

impl Material {
    pub fn aligned(&self) -> AlignedMaterial {
        // ray_tracer_shader::Material
        AlignedMaterial {
            albedo: self.albedo.to_array(),
            ty: self.ty as u32,
            fuzz: self.fuzz,
            refraction_ratio: self.refraction_ratio,
            _padding: [0; 2],
        }
    }

    pub fn new_lambertian(albedo: Vec3) -> Self {
        Self {
            albedo,
            ty: MaterialType::Lambertian,
            fuzz: 0.0,
            refraction_ratio: 0.0,
        }
    }

    pub fn new_metal(albedo: Vec3, fuzz: f32) -> Self {
        assert!(fuzz <= 1.0);
        Self {
            albedo,
            ty: MaterialType::Metal,
            fuzz,
            refraction_ratio: 0.0,
        }
    }

    pub fn new_dielectric(refraction_ratio: f32) -> Self {
        Self {
            albedo: Vec3::new(1.0, 1.0, 1.0),
            ty: MaterialType::Dielectric,
            fuzz: 0.0,
            refraction_ratio,
        }
    }

    pub fn new_light(luminance: Vec3) -> Self {
        // println!("Luminance: {:?}", luminance);
        Self {
            albedo: luminance,
            ty: MaterialType::Light,
            fuzz: 0.0,
            refraction_ratio: 0.0,
        }
    }
}