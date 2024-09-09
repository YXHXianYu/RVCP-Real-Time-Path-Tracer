
#[derive(Debug, Clone, Copy)]
pub struct Material {
    id: u32,
}

impl Material {
    pub fn matte() -> Self {
        Self {
            id: 0,
        }
    }

    pub fn to_shader(&self) -> u32 {
        self.id
    }
}