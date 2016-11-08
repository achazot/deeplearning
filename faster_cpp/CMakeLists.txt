cmake_minimum_required (VERSION 2.8)

project (faster_rcnn_demo)

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

set(CMAKE_BUILD_TYPE Debug)


# ------------------------------------------------------------------------------
# comp
# ------------------------------------------------------------------------------

add_executable(main main.cpp)


# ------------------------------------------------------------------------------
# libs
# ------------------------------------------------------------------------------

include_directories
(
	"${PROJECT_SOURCE_DIR}/../caffe-fast-rcnn/include"
	"${PROJECT_SOURCE_DIR}/../lib/nms"
	"${PROJECT_SOURCE_DIR}/../lib"
	/usr/local/include
	/usr/include/python2.7
	/home/dibi/sources/py-faster-rcnn/caffe-fast-rcnn/include
	/usr/include/boost/numeric/odeint/external/mkl
	/home/dibi/sources/py-faster-rcnn/lib/nms
	/usr/local/cuda-7.5/include
)

target_link_libraries
(
	main
	/home/dibi/sources/py-faster-rcnn/caffe-fast-rcnn/build/lib/libcaffe.so
	/home/dibi/sources/py-faster-rcnn/lib/nms/gpu_nms.so
	/usr/lib/x86_64-linux-gnu//libboost_system.so
	/usr/lib/x86_64-linux-gnu//libboost_python.so
	/usr/lib/x86_64-linux-gnu/libglog.so
	/usr/lib/x86_64-linux-gnu/libpython2.7.so
	/usr/local/lib/libopencv_core.so
	/usr/local/lib/libopencv_highgui.so
	/usr/local/lib/libopencv_imgproc.so
)
