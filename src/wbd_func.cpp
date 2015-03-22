#include "wbd_func.h"

namespace wbd
{
	void bindFloatImageTo2DTexture(cudaTextureObject_t* textureObject, float* image, uint32 width, uint32 height)
	{
		cudaResourceDesc resourceDesc;
		memset(&resourceDesc, 0, sizeof(resourceDesc));
		resourceDesc.resType = cudaResourceTypePitch2D;
		resourceDesc.res.pitch2D.devPtr = image;
		resourceDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
		resourceDesc.res.pitch2D.desc.x = 32; // bits per channel
		resourceDesc.res.pitch2D.width = width;
		resourceDesc.res.pitch2D.height = height;
		resourceDesc.res.pitch2D.pitchInBytes = width * sizeof(float);

		cudaTextureDesc textureDesc;
		memset(&textureDesc, 0, sizeof(textureDesc));
		textureDesc.readMode = cudaReadModeElementType;
		textureDesc.filterMode = cudaFilterModeLinear;
		textureDesc.addressMode[0] = cudaAddressModeClamp;
		textureDesc.addressMode[1] = cudaAddressModeClamp;
		textureDesc.normalizedCoords = false;

		cudaCreateTextureObject(textureObject, &resourceDesc, &textureDesc, NULL);
	}

	void bindLinearFloatDataToTexture(cudaTextureObject_t* textureObject, float* data, uint32 count)
	{
		cudaResourceDesc resourceDesc;
		memset(&resourceDesc, 0, sizeof(resourceDesc));
		resourceDesc.resType = cudaResourceTypeLinear;
		resourceDesc.res.linear.devPtr = data;
		resourceDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		resourceDesc.res.linear.desc.x = 32; // bits per channel
		resourceDesc.res.linear.sizeInBytes = count * sizeof(float);

		cudaTextureDesc textureDesc;
		memset(&textureDesc, 0, sizeof(textureDesc));
		textureDesc.readMode = cudaReadModeElementType;	

		cudaCreateTextureObject(textureObject, &resourceDesc, &textureDesc, NULL);
	}
}
