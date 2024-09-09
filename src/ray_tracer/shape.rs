use vulkano::buffer::BufferContents;

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: glam::Vec3,
    pub radius: f32,
    pub material_id: u32,
}

#[derive(Debug, Clone, Copy, BufferContents)]
#[repr(C)]
pub struct AlignedSphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub material_id: u32,
    pub _padding: [u32; 3],
}

impl Sphere {
    pub fn aligned(&self) -> AlignedSphere {
        // ray_tracer_shader::Sphere;
        AlignedSphere {
            center: self.center.to_array(),
            radius: self.radius,
            material_id: self.material_id,
            _padding: [0; 3],
        }
    }
}