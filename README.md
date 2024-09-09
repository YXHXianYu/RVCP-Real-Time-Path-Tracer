# VulkanLearning

My Vulkan Learning Repo.

The target is to build a real-time ray tracer using ~~vulkan~~ **vulkano**.

## Preview

* Fractal (Refer to https://vulkano.rs/05-images/04-mandelbrot.html)
  * ![fractal](./README/fractal.png)

## 笔记：二阶段

* 实现一个基于Compute Shader的Ray Tracing

### 第一步：实现一个能在屏幕上绘制颜色的Compute Shader

* Compute Pipeline vs. Graphics Pipeline
  * 区别蛮大的
* 一个bug，找了两个小时
  * ![image-20240907203838722](./README/image-20240907203838722.png)
  * **操蛋啊！！！！**
* 搞定

### 第二步：设计Compute Shader光线追踪数据结构

* CPU端
* GPU端
  * 输入Buffer：Shapes、Material、Light
  * 输出Buffer：画布
  * PushConstants
    * 摄像机数据：按照MoerLite实现
      * Transform（Position、Up、LookAt）, t_near, t_far, vertical_f........
* 光线追踪流程
  * Camera：相机生成光线（Moer-Lite在一个像素点处进行多次采用，我们这里进行简化，直接根据像素中心点采样）—— 通过PushConstants传递数据，光线生成在shader中硬编码
  * Sampler：采样器 —— 在shader中硬编码
  * Integrator：积分器 —— 在shader中硬编码
  * Shape：几何体 —— 通过Buffer传输
  * Acceleration：加速结构 —— 先不考虑
  * Light：光源 —— 通过Buffer传输
  * Material：材质 —— 通过Buffer传输

### 第三步：往shader中导入数据

* **大坑**：vulkano-shaders会对shader代码进行解析。而entry point只会包含 **编译器优化后的代码中 使用到的变量信息**。也就是说，如果你定义了一些binding point，但是main()函数中不会使用到这些binding point，那么你就无法在descriptor point中对其进行绑定！会导致运行时错误！
  * 所以，以后最好先写shader的雏形，再往里导入数据

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
  * ImageView有两种创建方式
    1. 从Swapchain中获取Image
    2. 自己创建Image

### RenderPass

* 获取Subpass需要从RenderPass中得到
  * 类似Queue和QueueFamily、Layout和SetLayouts的关系

### Swapchain

* Instance依赖Vulkan Library
* Surface依赖Instance和Window（Window需要由winit等库创建）
* Physical Device依赖于Instance选择（从Instance访问机器上所有GPU）
* Device依赖Physical Device中取出的Queue Family Index
* 最后，Swapchain依赖Device和Surface



