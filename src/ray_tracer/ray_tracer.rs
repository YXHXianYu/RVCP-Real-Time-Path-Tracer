
use std::collections::HashMap;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use super::scene::Camera;
use super::vulkan::{RuntimeInfo, Vk};

/**
 * TODO: 
 * 下一步重构：vk和info合并；
 * ===> Camera数据完毕
 * 开始准备Shape, Material, Light数据
 */


pub fn run() {
    println!("Running ray tracer");
    let (vk, info, event_loop) = Vk::new();
    main_loop(vk, info, event_loop);
}

fn main_loop(mut vk: Vk, mut info: RuntimeInfo, event_loop: EventLoop<()>) {

    const TIME_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                info.is_window_resized = true;
            },
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(key),
                            state,
                            ..
                        },
                    ..
                },
                ..
            } => {
                let is_pressed = state == ElementState::Pressed;
                info.keyboard_is_pressing.insert(key, is_pressed);
            },
            // 捕获鼠标位移差值
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                info.mouse_cur_position = position.into();
            },
            Event::MainEventsCleared => {
                info.fps_frame_count += 1;
                let now = std::time::Instant::now();
                if now - info.fps_last_time >= TIME_INTERVAL {
                    println!("Rendering FPS: {}", info.fps_frame_count);
                    info.fps_frame_count = 0;
                    info.fps_last_time = now;
                }

                let delta_time = (std::time::Instant::now() - info.last_tick_time).as_secs_f64() as f32;
                info.last_tick_time = now; // for other events

                update_camera_state(
                    &mut info,
                    delta_time,
                );

                vk.update_frame(&mut info);
            },
            _ => (),
        }
    });
}

fn update_camera_state(
    info: &mut RuntimeInfo,
    delta_time: f32,
) {
    macro_rules! f {
        ($key:expr, $stmt:stmt) => {
            if info.keyboard_is_pressing.get(&$key).is_some_and(|x| *x) {
                info.is_new_push_constants = true;
                $stmt
            }
        };
    }

    let camera = &mut info.scene.camera;
    let move_v = camera.move_speed * delta_time;
    let rotate_v = camera.rotate_speed * delta_time;

    println!("camera_right: {:?}", camera.right);

    f!(VirtualKeyCode::A, { camera.position -= camera.right * move_v; });
    f!(VirtualKeyCode::D, { camera.position += camera.right * move_v; });
    f!(VirtualKeyCode::S, { camera.position -= camera.forward * move_v; });
    f!(VirtualKeyCode::W, { camera.position += camera.forward * move_v; });
    f!(VirtualKeyCode::Q, { camera.position -= camera.up * move_v; });
    f!(VirtualKeyCode::E, { camera.position += camera.up * move_v; });

    if info.mouse_last_position.0 >= 0.0 {
        let dx = info.mouse_cur_position.0 - info.mouse_last_position.0;
        let dy = info.mouse_cur_position.1 - info.mouse_last_position.1;
        camera.yaw += dx * rotate_v;
        camera.pitch -= dy * rotate_v;
        camera.pitch = camera.pitch.clamp(-89.0, 89.0);

        camera.forward = glam::Vec3::new(
            camera.yaw.to_radians().cos() * camera.pitch.to_radians().cos(),
            camera.pitch.to_radians().sin(),
            camera.yaw.to_radians().sin() * camera.pitch.to_radians().cos(),
        ).normalize();
        camera.right = camera.forward.cross(glam::Vec3::Y).normalize();
    }
    info.mouse_last_position = info.mouse_cur_position;

}
