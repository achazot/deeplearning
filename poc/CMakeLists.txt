cmake_minimum_required (VERSION 2.6)
include(FindPackageHandleStandardArgs)

project (poc)

option(CAFFE_CPU_ONLY "Use CPU only for Caffe" OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# ------------------------------------------------------------------------------
# C++11 settings
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

find_package(Boost 1.40 COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(poc ${Boost_LIBRARIES})

include("cmake/External/glog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})

find_package(OpenCV CONFIG REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(poc ${OpenCV_LIBRARIES})

cmake_policy(PUSH)
cmake_policy(SET CMP0012 OLD)
find_package(Caffe CONFIG REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
target_link_libraries(poc ${Caffe_LIBRARIES})
cmake_policy(POP)


if(CPU_ONLY)
  add_definitions(-DCPU_ONLY)
endif()
