
use glam::Vec3;
use winit::dpi::{LogicalPosition, PhysicalSize};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

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

    // vk.window.set_inner_size(PhysicalSize::new(256, 256));
    vk.window.set_inner_size(PhysicalSize::new(384, 384));
    // vk.window.set_inner_size(PhysicalSize::new(512, 512));
    // vk.window.set_inner_size(PhysicalSize::new(1024, 1024));
    vk.window_size = vk.window.inner_size();

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
            // 捕获鼠标右键是否按下
            Event::WindowEvent {
                event: WindowEvent::MouseInput { state, button, .. },
                ..
            } => {
                if button == winit::event::MouseButton::Right {
                    info.is_mouse_right_button_pressing = state == ElementState::Pressed;
                    info.mouse_cur_position = ((vk.window_size.width / 2) as f32, (vk.window_size.height / 2) as f32);
                }
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
                    &mut vk,
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
    vk: &mut Vk,
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

    f!(VirtualKeyCode::A, { camera.position -= camera.right * move_v; });
    f!(VirtualKeyCode::D, { camera.position += camera.right * move_v; });
    f!(VirtualKeyCode::S, { camera.position -= camera.forward * move_v; });
    f!(VirtualKeyCode::W, { camera.position += camera.forward * move_v; });
    f!(VirtualKeyCode::Q, { camera.position -= Vec3::Y * move_v; });
    f!(VirtualKeyCode::E, { camera.position += Vec3::Y * move_v; });

    let window_center = (vk.window_size.width / 2, vk.window_size.height / 2);
    // println!("mouse_pos: {:?}; window_center: {:?}", info.mouse_cur_position, window_center);

    if info.is_mouse_right_button_pressing {
        info.is_new_push_constants = true;

        let dx = info.mouse_cur_position.0 - window_center.0 as f32;
        let dy = info.mouse_cur_position.1 - window_center.1 as f32;
        camera.yaw += dx * rotate_v;
        camera.pitch -= dy * rotate_v;
        camera.pitch = camera.pitch.clamp(-89.0, 89.0);

        let yaw_rad = camera.yaw.to_radians();
        let pitch_rad = camera.pitch.to_radians();
        camera.forward = Vec3::new(
            yaw_rad.cos() * pitch_rad.cos(),
            pitch_rad.sin(),
            yaw_rad.sin() * pitch_rad.cos(),
        ).normalize();
        camera.right = camera.forward.cross(Vec3::Y).normalize();
        camera.up = camera.right.cross(camera.forward).normalize();
    }

    if info.is_mouse_right_button_pressing {
        // 隐藏且锁定鼠标
        vk.window.set_cursor_position(LogicalPosition::new(
            window_center.0,
            window_center.1
        )).unwrap();
        vk.window.set_cursor_visible(false);
    } else {
        // 显示且解锁鼠标
        vk.window.set_cursor_visible(true);
    }

}
