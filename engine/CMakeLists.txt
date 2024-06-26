# Vulkan
set(vulkan_include ${THIRD_PARTY_DIR}/VulkanSDK/include)

if(WIN32)
    set(vulkan_lib ${THIRD_PARTY_DIR}/VulkanSDK/lib/Win32/vulkan-1.lib)
elseif(UNIX)
    if(APPLE)
      set(vulkan_lib ${THIRD_PARTY_DIR}/VulkanSDK/lib/MacOS/libvulkan.1.dylib)
    else()
      set(vulkan_lib ${THIRD_PARTY_DIR}/VulkanSDK/lib/Linux/libvulkan.so.1)
    endif()
else()
  message(FATAL_ERROR "Unknown Platform")
endif()

# Shader
set(SHADER_COMPILE_TARGET MyShader)
add_subdirectory(shader)

# 3rdparty
add_subdirectory(3rdparty)

# Runtime
add_subdirectory(source)