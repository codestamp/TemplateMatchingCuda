cmake_minimum_required(VERSION 2.8)

	project(GpuTemplateMatch)
	find_package(OpenCV REQUIRED)
	find_package(CUDA)
# nvcc flags

	set(
			CUDA_NVCC_FLAGS 
			${CUDA_NVCC_FLAGS};
			-Wno-deprecated-gpu-targets
			-gencode arch=compute_20,code=sm_20
	   )
cuda_compile(TESTIT_O source/test.cu)
# include directories
#include_directories(
#		includes
#		)

	cuda_add_executable(1test ${TESTIT_O})
target_link_libraries(1test ${OpenCV_LIBS})
