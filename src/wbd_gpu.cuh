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
	} // namespace gpu
} // namespace wbd

#endif // CUH_WBD_GPU
