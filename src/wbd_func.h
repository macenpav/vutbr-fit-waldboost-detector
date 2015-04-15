/**
 * @file	wbd_func.h
 * @brief Waldboost detector functions shared by all the implementations.
 *
 * @author Pavel Macenauer <macenauer.p@gmail.com>
 */

#ifndef H_WBD_FUNC
#define H_WBD_FUNC

#include "wbd_general.h"
#include <cuda_runtime.h>
#include <iostream>

#define GPU_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::cerr << "GPU assert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;		
		if (abort) exit(code);
	}
}

namespace wbd
{
	void bindFloatImageTo2DTexture(cudaTextureObject_t* textureObject, float* image, uint32 width, uint32 height);

	void bindLinearFloatDataToTexture(cudaTextureObject_t* textureObject, float* data, uint32 count);
}

#endif
