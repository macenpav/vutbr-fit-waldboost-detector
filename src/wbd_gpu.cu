#include "wbd_gpu.cuh"
#include "wbd_structures.h"

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

		namespace pyramid
		{
			__global__ void createFirstPyramid(float* outImage, float* finalImage, cudaTextureObject_t inImageTexture, const uint32 pyramidWidth, const uint32 pyramidHeight, const uint32 imageWidth, const uint32 imageHeight, const uint32 pitch, const uint32 levels)
			{
				const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
				const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;
				
				if (x < pyramidWidth && y < pyramidHeight)
				{
					const float scalingFactor = __powf(2, 1 / static_cast<float>(levels));					
					uint32 currentHeight = imageHeight, currentCumulHeight = imageHeight, currentWidth = imageWidth, currentOffsetY = 0;
					for (uint8 level = 0; level < levels; ++level)
					{																										
						if (y < currentCumulHeight)
							break;										

						currentOffsetY += currentHeight;
						currentWidth = static_cast<uint32>(static_cast<float>(currentWidth) / scalingFactor);
						currentHeight = static_cast<uint32>(static_cast<float>(currentHeight) / scalingFactor);
						currentCumulHeight += currentHeight;						
					}
		
					float result;
					// black if its outside of the image
					// or downsample from the original texture
					if (x > currentWidth)
						result = 0.f;
					else
					{
						float origX = static_cast<float>(x) / static_cast<float>(currentWidth) * static_cast<float>(imageWidth);
						float origY = static_cast<float>(y - currentOffsetY) / static_cast<float>(currentHeight) * static_cast<float>(imageHeight);
						result = tex2D<float>(inImageTexture, origX, origY);
					}

					outImage[y * imageWidth + x] = result;
					finalImage[y * pitch + x] = result;
				}
			}

			__global__ void createPyramidFromPyramid(float* outImage, float* finalImage, cudaTextureObject_t inImageTexture, const uint32 width, const uint32 height, const uint32 offsetX, const uint32 offsetY, const uint32 pitch)
			{
				const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
				const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;				

				if (x < width && y < height)
				{
					// x,y in previous octave
					const float prevX = static_cast<float>(x)* 2.f;
					const float prevY = static_cast<float>(y)* 2.f;

					float res = tex2D<float>(inImageTexture, prevX, prevY);

					outImage[y * width + x] = res;
					finalImage[(y + offsetY) * pitch + (x + offsetX)] = res;
				}
			}		
		} // namespace pyramid
	} // namespace gpu
} // namespace wbd
