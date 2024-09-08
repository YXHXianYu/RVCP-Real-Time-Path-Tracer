
use std::collections::HashMap;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use super::config::Config;
use super::vulkan::{RuntimeInfo, Vk};

pub fn run() {
    println!("Running ray tracer");
    let (vk, info, event_loop) = Vk::new(Config::default());
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
            }
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

                update_keyboard_state(
                    &mut info.keyboard_is_pressing,
                    &mut vk.config,
                    &mut info.is_new_push_constants,
                    delta_time,
                );

                vk.update_frame(&mut info);
            },
            _ => (),
        }
    });
}

fn update_keyboard_state(
    keyboard_is_pressing: &mut HashMap<VirtualKeyCode, bool>,
    config: &mut Config,
    is_new_push_constants: &mut bool,
    delta_time: f32,
) {
    macro_rules! f {
        ($key:expr, $stmt:stmt) => {
            if keyboard_is_pressing.get(&$key).is_some_and(|x| *x) {
                *is_new_push_constants = true;
                $stmt
            }
        };
    }

    let g = |x: f32| {
        x.powf(1.2).max(1.0)
    };

    let h = |x| {
        1.0_f32 / x
    };

    let pos_v = h(config.camera_scale.abs()) * config.camera_move_speed * delta_time;
    let scale_v = g(config.camera_scale.abs()) * config.camera_move_speed * delta_time;

    f!(VirtualKeyCode::A, config.camera_position[0] -= pos_v);
    f!(VirtualKeyCode::D, config.camera_position[0] += pos_v);
    f!(VirtualKeyCode::W, config.camera_position[1] -= pos_v);
    f!(VirtualKeyCode::S, config.camera_position[1] += pos_v);
    f!(VirtualKeyCode::Q, config.camera_scale -= scale_v);
    f!(VirtualKeyCode::E, config.camera_scale += scale_v);
}
