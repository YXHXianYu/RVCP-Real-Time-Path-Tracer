use vulkano::buffer::BufferContents;

use crate::ray_tracer_deprecated::prelude::*;

// MARK: Vertex

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
}

#[derive(Debug, Clone, Copy, BufferContents)]
#[repr(C)]
pub struct AlignedVertex {
    position: [f32; 4],
    normal: [f32; 4],
}

impl Vertex {
    pub fn aligned(&self) -> AlignedVertex {
        AlignedVertex {
            position: vec3_to_f32_4(self.position),
            normal: vec3_to_f32_4(self.normal),
        }
    }
}

// MARK: Face

#[derive(Debug, Clone, Copy)]
pub struct Face {
    pub vertices: glam::UVec3,
    pub material_id: u32,
}

#[derive(Debug, Clone, Copy, BufferContents)]
#[repr(C)]
pub struct AlignedFace {
    vertices: [u32; 3],
    material_id : u32,
}

impl Face {
    pub fn aligned(&self) -> AlignedFace {
        AlignedFace {
            vertices: self.vertices.to_array(),
            material_id: self.material_id,
        }
    }
}

// MARK: Mesh

#[derive(Debug)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub faces: Vec<Face>,
}