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
			/** @brief Creates a pyramid image from an image.
			 *
			 * @details Downsamples an image creating a pyramid image.
			 *
			 * @param outImage			Output pyramid.
			 * @param finalImage		Image canvas with multiple pyramids, used for detection.
			 * @param inImageTexture	Input image texture.
			 * @param pyramidWidth		Created pyramid width.
			 * @param pyramidHeight		Created pyramid height.
			 * @param imageWidth		Input image width.
			 * @param imageHeight		Input image height.
			 * @param pitch				Final image pitch.
			 * @param levels			Number of downsampled images inside pyramid.
			 * @return
			 */
			__global__ void createFirstPyramid(float* outImage, float* finalImage, cudaTextureObject_t inImageTexture, const uint32 pyramidWidth, const uint32 pyramidHeight, const uint32 imageWidth, const uint32 imageHeight, const uint32 pitch, const uint32 levels);

			/** @brief Downsamples a pyramid image.
			 *
			 * @details Creates a pyramid image by downsampling another pyramid image
			 *			passed as a texture.
			 *
			 * @param outImage			Downsampled pyramid.
			 * @param finalImage		Image canvas with multiple pyramids used for detection.
			 * @param inImageTexture	Pyramid input texture.
			 * @param width				Created pyramid width.
			 * @param height			Created pyramid height.
			 * @param offsetX			X-offset inside final image.
			 * @param offsetY			Y-offset inside final image.
			 * @param pitch				Final image pitch.
			 * @return					Void.
			 */
			__global__ void createPyramidFromPyramid(float* outImage, float* finalImage, cudaTextureObject_t inImageTexture, const uint32 width, const uint32 height, const uint32 offsetX, const uint32 offsetY, const uint32 pitch);
		} // namespace pyramid	
	} // namespace gpu
} // namespace wbd

#endif // CUH_WBD_GPU_PYRAMID
