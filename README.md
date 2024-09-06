# VulkanLearning

My Vulkan Learning Repo.

The target is to build a real-time ray tracer using ~~vulkan~~ **vulkano**.

## Preview

* Fractal (Refer to https://vulkano.rs/05-images/04-mandelbrot.html)
  * ![fractal](./README/fractal.png)

## 笔记

### Queue & QueueFamily

* GPU的队列&队列家族，可以类比成CPU的线程&线程池

* 每个不同的队列家族，会提供不同的操作类型。比如Graphics Queue Family，就提供一组（例如16个）队列，每个队列性质类似，可以执行Graphics相关的命令

* ![image-20240906155653114](./README/image-20240906155653114.png)

* 小尝试

  * 创建Device的时候，我们要给这个Device分配一个或多个Queue。这个Queue的数量是要预先指定的，否则就可能出现`queues.next().unwrap()`失败的情况

  * ```rust
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                queues: vec![0.5, 0.5],
                ..Default::default()
            }],
            ..Default::default()
        }
    ).unwrap();
    let queue = queues.next().unwrap();
    let queue2 = queues.next().unwrap();
    ```

* Queue具体在什么时候使用的呢？

  1. 创建Command Buffer的时候，需要传入该Queue的QueueFamilyIndex
  2. Submit Command的时候，需要传入Queue

* 【发现一个问题：创建CommandBuffer时，使用family index 0，但execute的时候，使用family index 1的一个queue实例。结果竟然是对的】

  * ![image-20240906161526216](./README/image-20240906161526216.png)
  * ![image-20240906161516061](./README/image-20240906161516061.png)

### Buffer

* 合法
  * ![image-20240906094903836](./README/image-20240906094903836.png)
  * 第二个 `buffer.write()` 改成 `buffer.read()`，仍合法
* 不合法
  * ![image-20240906094946006](./README/image-20240906094946006.png)
  * 原因有二：① 读写锁冲突；② Memory对CPU不可见

### Command Buffer

* 每个Command Buffer都需要和一个QueueFamily绑定
* 在Submission的时候，一个CommandBuffer才会和具体的Queue绑定

### Pipeline

* 如何创建一个Pipeline？
  * 一个Pipeline依赖于Stages和Layout（其中，Layout也依赖于Stages）
  * Stages包含了Pipeline的阶段信息，比如是否包含vertex shader和fragment shader

### Allocator

* 目前了解到的，总共有3种常见的Allocator，这些Allocator都可以复用，不需要每次新建
  * MemoryAllocator
  * CommandBufferAllocator
  * DescriptorSetAllocator
* Image和Buffer创建时参数的对比
  * Buffer::from_iter() 有4个参数
    * memory allocator、buffer create info、allocation info、initial data
  * Image::new() 有3个参数
    * memory allocator、image create info、allocation info
  * 其中，第1个和第3个参数都是一样的

### Framebuffer

* Framebuffer的创建依赖RenderPass和ImageView
  * 【Framebuffer的attachments信息，是否要和RenderPass的attachments信息一致？】

### RenderPass

* 获取Subpass需要从RenderPass中得到
  * 类似Queue和QueueFamily、Layout和SetLayouts的关系



