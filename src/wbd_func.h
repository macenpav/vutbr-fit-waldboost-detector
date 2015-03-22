/**
* @file	wb_general.h
* @brief Waldboost detector functions shared by all the implementations.
*
* @author Pavel Macenauer <macenauer.p@gmail.com>
*/

#ifndef H_WBD_FUNC
#define H_WBD_FUNC

#include "wbd_general.h"
#include "cuda_runtime.h"

namespace wbd
{
	void bindFloatImageTo2DTexture(cudaTextureObject_t* textureObject, float* image, uint32 width, uint32 height);

	void bindLinearFloatDataToTexture(cudaTextureObject_t* textureObject, float* data, uint32 count);
}

#endif