# set directory
set(TINY_OBJ_LOADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader)

# add an interface library
add_library(tinyobjloader INTERFACE)

# add include directory
target_include_directories(tinyobjloader INTERFACE "${TINY_OBJ_LOADER_DIR}")