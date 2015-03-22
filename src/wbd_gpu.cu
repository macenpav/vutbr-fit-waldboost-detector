#include "wbd_gpu.cuh"

namespace wbd
{
	namespace gpu
	{
		__global__ void preprocess(float* preprocessedImage, uint8* image, const uint32 width, const uint32 height)
		{
			const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
			const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

			if (x < width && y < height)
			{
				const uint32 i = y * width + x;
				// convert to B&W
				preprocessedImage[i] = WB_RGB2BW_RED * static_cast<float>(image[3 * i])
					+ WB_RGB2BW_GREEN * static_cast<float>(image[3 * i + 1])
					+ WB_RGB2BW_BLUE * static_cast<float>(image[3 * i + 2]);

				// clip to <0.0f;1.0f>
				preprocessedImage[i] /= 255.0f;
			}
		}		
	} // namespace gpu
} // namespace wbd
