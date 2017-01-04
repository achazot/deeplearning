cmake_minimum_required (VERSION 2.8)

project (FasTrack)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)


# ------------------------------------------------------------------------------
# create file tree
# ------------------------------------------------------------------------------

file(MAKE_DIRECTORY bin)
file(MAKE_DIRECTORY input)
file(MAKE_DIRECTORY output)
file(MAKE_DIRECTORY networks)

# ------------------------------------------------------------------------------
# compiler settings
# ------------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
	if(NOT CMAKE_CXX_FLAGS MATCHES "-std=c\\+\\+11")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	endif()
endif()
set(CMAKE_CXX_EXTENSIONS 0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES -DNO_STRICT)
add_definitions(-D"USE_OPENCV=1")


find_package(CUDA)
if (CUDA_FOUND)
	message( "--   -> Using GPU" )
else ()
	message( "--   -> Using CPU ONLY" )
	add_definitions(-DCPU_ONLY)
endif ()

set(CMAKE_BUILD_TYPE Debug)

# ------------------------------------------------------------------------------
# compilation
# ------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCES "src/*.cpp")

set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

add_executable (
	fastrack ${SOURCES}
)

# ------------------------------------------------------------------------------
# libs
# ------------------------------------------------------------------------------

find_package(Boost 1.40 COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(fastrack ${Boost_LIBRARIES})

include("cmake/External/glog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})

find_package(OpenCV CONFIG REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(fastrack ${OpenCV_LIBRARIES})

include_directories (
	"${PROJECT_SOURCE_DIR}/../caffe-fast-rcnn/include"
	"${PROJECT_SOURCE_DIR}/../lib/nms"
	"${PROJECT_SOURCE_DIR}/../lib"
	/usr/local/include
	/usr/include/python2.7
	/home/dibi/sources/py-faster-rcnn/caffe-fast-rcnn/include
	/usr/include/boost/numeric/odeint/external/mkl
	/home/dibi/sources/py-faster-rcnn/lib/nms
	/usr/local/cuda-8.0/include
)

target_link_libraries (
	fastrack
	/home/dibi/sources/py-faster-rcnn/caffe-fast-rcnn/build/lib/libcaffe.so
	/home/dibi/sources/py-faster-rcnn/lib/nms/gpu_nms.so
	/usr/lib/x86_64-linux-gnu/libglog.so
	/usr/lib/x86_64-linux-gnu/libpython2.7.so
)
