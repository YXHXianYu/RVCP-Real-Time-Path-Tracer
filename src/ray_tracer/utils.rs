use glam::Vec3;


#[allow(dead_code)]
pub fn vec3_to_f32_4(vec3: Vec3) -> [f32; 4] {
    [vec3[0], vec3[1], vec3[2], 0.0]
}