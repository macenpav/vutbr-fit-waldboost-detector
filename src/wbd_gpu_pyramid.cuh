/**
 * @file	wbd_gpu_pyramid.cuh
 * @brief Waldboost detector gpu functions for pyramid generation.
 *
 * @author Pavel Macenauer <macenauer.p@gmail.com>
 */

#ifndef CUH_WBD_GPU_PYRAMID
#define CUH_WBD_GPU_PYRAMID

#include "wbd_general.h"
#include "cuda_runtime.h"

namespace wbd
{
	namespace gpu
	{
		namespace pyramid
		{
			/** @brief
			 *
			 * @details
			 *
			 * @param outImage
			 * @param finalImage
			 * @param inImageTexture
			 * @param pyramidWidth
			 * @param pyramidHeight
			 * @param imageWidth
			 * @param imageHeight
			 * @param pitch
			 * @param levels
			 * @return
			 */
			__global__ void createFirstPyramid(float* outImage, float* finalImage, cudaTextureObject_t inImageTexture, const uint32 pyramidWidth, const uint32 pyramidHeight, const uint32 imageWidth, const uint32 imageHeight, const uint32 pitch, const uint32 levels);

			/** @brief
			 *
			 * @details
			 *
			 * @param outImage
			 * @param finalImage
			 * @param inImageTexture
			 * @param width
			 * @param height
			 * @param offsetX
			 * @param offsetY
			 * @param pitch
			 * @return
			 */
			__global__ void createPyramidFromPyramid(float* outImage, float* finalImage, cudaTextureObject_t inImageTexture, const uint32 width, const uint32 height, const uint32 offsetX, const uint32 offsetY, const uint32 pitch);
		} // namespace pyramid	
	} // namespace gpu
} // namespace wbd

#endif // CUH_WBD_GPU_PYRAMID
