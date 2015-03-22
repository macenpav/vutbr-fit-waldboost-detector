/**
* @file	wbd_gpu.cuh
* @brief Waldboost detector general gpu functions.
*
* @author Pavel Macenauer <macenauer.p@gmail.com>
*/

#ifndef CUH_WBD_GPU
#define CUH_WBD_GPU

#include "wbd_general.h"
#include "cuda_runtime.h"

namespace wbd
{
	namespace gpu
	{
		/** @brief Preprocessing kernel.
		 *
		 * @details GPU kernel doing preprocessing - converts image to black and white and integer
		 * values to float.	FP values are then stored as a texture.
		 *
		 * @param preprocessedImage	Output image.
		 * @param image				Input image.
		 * @param width				Image width.
		 * @param height				Image height.
		 * @return					Void.
		 */
		__global__ void preprocess(float* preprocessedImage, uint8* image, const uint32 width, const uint32 height);	

		/** @brief Clears images, sets all pixels to black. 
		 *		 
		 *
		 * @param imageData	Image data.
		 * @param width		Image width.
		 * @param height	Image height.
		 * @return Void.
		 */
		__global__ void clearImage(float* imageData, const uint32 width, const uint32 height);

		/** @brief Copies an image from texture to gpu memory. 
		 *
		 * @param imageData	Memory pointer to GPU image data.
		 * @param texture	Texture.
		 * @param width		Image width.
		 * @param height	Image height.
		 */
		__global__ void copyImageFromTextureObject(float* imageData, cudaTextureObject_t texture, const uint32 width, const uint32 height);

	} // namespace gpu
} // namespace wbd

#endif // CUH_WBD_GPU
