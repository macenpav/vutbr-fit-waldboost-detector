#include "wb_waldboostdetector.h"

#include <iostream>

namespace wb {	

	__global__ void preparationKernel(uint8* inData, float* outData)
	{
		uint32 threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId < DEV_INFO.imageSize)
		{
			// convert to B&W
			outData[threadId] = 0.299f * static_cast<float>(inData[3 * threadId])
				+ 0.587f * static_cast<float>(inData[3 * threadId + 1])
				+ 0.114f * static_cast<float>(inData[3 * threadId + 2]);

			// clip to <0.0f;1.0f>
			outData[threadId] /= 255.0f;
		}			
	}

	void WaldboostDetector::init(cv::Mat* image)
	{
		_info.width = image->cols;
		_info.height = image->rows;
		_info.imageSize = image->cols * image->rows;		
		_info.channels = image->channels();

		cudaMemcpyToSymbol(devInfo, &_info, sizeof(ImageInfo));
		cudaMalloc((void**)&_devOriginalImage, sizeof(uint8) * _info.imageSize * _info.channels);
		cudaMalloc((void**)&_devWorkingImage, sizeof(float) * _info.imageSize);
	}

	void WaldboostDetector::setImage(cv::Mat* image)
	{			
		cudaMemcpy(_devOriginalImage, image->data, _info.imageSize * _info.channels * sizeof(uint8), cudaMemcpyHostToDevice);
		
		dim3 grid(4096, 1, 1);
		dim3 block(1024, 1, 1);

		preparationKernel <<<grid, block >>>(_devOriginalImage, _devWorkingImage);
				
		cv::Mat tmp(cv::Size(_info.width, _info.height), CV_32FC1);
		
		cudaMemcpy(tmp.data, _devWorkingImage, _info.imageSize * sizeof(float), cudaMemcpyDeviceToHost);
		cv::imshow("converted to float and bw", tmp);
		cv::waitKey(WAIT_DELAY);
	}

	void WaldboostDetector::run()
	{

	}

	void WaldboostDetector::free()
	{
		cudaFree(_devOriginalImage);
		cudaFree(_devWorkingImage);
	}

	void WaldboostDetector::buildPyramid()
	{

	}

	void WaldboostDetector::bilinearInterpolation()
	{

	}
}
