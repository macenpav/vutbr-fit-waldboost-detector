project(WaldboostDetector)
cmake_minimum_required(VERSION 2.8)
set(WBD_VERSION_MAJOR 1)
set(WBD_VERSION_MINOR 0)

# ---------------------------------------------
# LIBRARIES
# ---------------------------------------------
# OPENCV
if (WIN32)
	set(OpenCV_DIR "C:/dev/libs/opencv249/build32")				
else ()
	set(OpenCV_DIR "/usr/lib/opencv")		
endif ()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_30,code=sm_30) 
# ---------------------------------------------


# ---------------------------------------------
# Sources
file(GLOB local_sources *.cpp *.h *.cuh *.cu)	
set (WBD_SOURCES ${local_sources})

cuda_add_executable(WaldboostDetector ${WBD_SOURCES})
target_link_libraries(WaldboostDetector ${OpenCV_LIBS})
# ---------------------------------------------


# ---------------------------------------------
# Install options
if (WIN32)
	set(INSTALL_DIR ${PROJECT_BINARY_DIR}/../bin)
else ()
	# whatever options you want
endif ()

install(TARGETS WaldboostDetector DESTINATION ${INSTALL_DIR})
# ---------------------------------------------