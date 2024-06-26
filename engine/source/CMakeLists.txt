
# Target Name
set(TARGET_NAME BJTU-Game-Engine)

# Source file
file(GLOB_RECURSE SOURCE_FILE CONFIGURE_DEPENDS "*.cpp")

if(BUILD_RUNTIME_TO_EXECUTABLE)
  add_executable(${TARGET_NAME} ${SOURCE_FILE})
else()
  list(FILTER SOURCE_FILE EXCLUDE REGEX "main\\.cpp$")
  add_library(${TARGET_NAME} ${SOURCE_FILE})
endif()

# Include file
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_include_directories(
  ${TARGET_NAME}
  PUBLIC $<BUILD_INTERFACE:${vulkan_include}>
)
target_include_directories(
  ${TARGET_NAME} 
  PUBLIC $<BUILD_INTERFACE:${THIRD_PARTY_DIR}/vulkanmemoryallocator/include>
)

# Link
target_link_libraries(${TARGET_NAME} PUBLIC fmt::fmt)
target_link_libraries(${TARGET_NAME} PUBLIC glfw)
target_link_libraries(${TARGET_NAME} PUBLIC glm)
target_link_libraries(${TARGET_NAME} PUBLIC stb)
target_link_libraries(${TARGET_NAME} PUBLIC ${vulkan_lib})

# Add Shader Dependency
add_dependencies(${TARGET_NAME} ${SHADER_COMPILE_TARGET})
target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${ENGINE_ROOT_DIR}/shader/generated/cpp>)

# === Post build commands (begin) ===
set(POST_BUILD_COMMANDS
  COMMAND ${CMAKE_COMMAND} -E make_directory "${BINARY_ROOT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>/" "${BINARY_ROOT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${BINARY_ROOT_DIR}/${ENGINE_ASSET_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${ENGINE_ROOT_DIR}/${ENGINE_ASSET_DIR}" "${BINARY_ROOT_DIR}/${ENGINE_ASSET_DIR}"
)
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD ${POST_BUILD_COMMANDS})
# === Post build commands (end) ===