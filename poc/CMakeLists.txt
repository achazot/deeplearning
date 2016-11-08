cmake_minimum_required (VERSION 2.6)
include(FindPackageHandleStandardArgs)

project (poc)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# ------------------------------------------------------------------------------
# settings
# ------------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-std=c\\+\\+11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
endif()


set(CMAKE_CXX_EXTENSIONS 0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)

add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES -DNO_STRICT)
add_definitions(-D"USE_OPENCV=1")


find_package(CUDA)
if (CUDA_FOUND)
	message( "Using GPU" )
else ()
	message( "Using CPU ONLY" )
  add_definitions(-DCPU_ONLY)
endif ()


# ------------------------------------------------------------------------------
# Actual compilation
# ------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCES "src/*.cpp")

set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

add_executable(
	poc ${SOURCES}
)


# ------------------------------------------------------------------------------
# Libs
# ------------------------------------------------------------------------------


cmake_policy(PUSH)
cmake_policy(SET CMP0012 OLD)
find_package(Caffe CONFIG REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
target_link_libraries(poc ${Caffe_LIBRARIES})
cmake_policy(POP)
