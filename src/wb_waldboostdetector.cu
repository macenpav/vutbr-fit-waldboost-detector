#include "wb_waldboostdetector.h"

#include <iostream>

#include "cuda_runtime.h"

#include "wb_general.h"
#include "wb_structures.h"
#include "wb_detector.h"
#include "wb_alphas.h"

namespace wb {
		
	__device__ 
	void detectDetections(		
		SurvivorData*	survivors,
		Detection*		detections,
		uint32*			detectionCount,
		uint16			startStage)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;		
		const uint32 globalId = y * PYRAMID.width + x;

		if (x < (PYRAMID.width - CLASSIFIER_WIDTH) && y < (PYRAMID.height - CLASSIFIER_HEIGHT))
		{
			float response = survivors[globalId].response;

			if (response < FINAL_THRESHOLD)
				return;

			const uint32 x = survivors[globalId].x;
			const uint32 y = survivors[globalId].y;

			__syncthreads();

			bool survived = eval(x, y, &response, startStage, STAGE_COUNT);
			if (survived) {
				uint32 pos = atomicInc(detectionCount, MAX_DETECTIONS);
				detections[pos].x = x;
				detections[pos].y = y;
				detections[pos].width = CLASSIFIER_WIDTH;
				detections[pos].height = CLASSIFIER_HEIGHT;
				detections[pos].response = response;
			}
		}
	}

	__device__ 
	void detectSurvivorsInit(		
		SurvivorData*	survivors,
		uint16			endStage)
	{
		__shared__ uint32 localSurvivors[BLOCK_SIZE];

		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;
		const uint32 blockId = threadIdx.y * blockDim.x + threadIdx.x;
		const uint32 globalId = y * PYRAMID.width + x;

		localSurvivors[blockId] = 0;

		if (x < (PYRAMID.width - CLASSIFIER_WIDTH) && y < (PYRAMID.height - CLASSIFIER_HEIGHT)) 
		{			
			float response = 0.0f;
			bool survived = eval(x, y, &response, 0, endStage);

			localSurvivors[blockId] = static_cast<uint32>(survived);

			// up-sweep
			uint32 offset = 1;
			for (uint32 d = BLOCK_SIZE >> 1; d > 0; d >>= 1, offset <<= 1) {
				__syncthreads();

				if (blockId < d)
				{
					uint32 ai = offset * (2 * blockId + 1) - 1;
					uint32 bi = offset * (2 * blockId + 2) - 1;
					localSurvivors[bi] += localSurvivors[ai];
				}
			}

			// down-sweep
			if (blockId == 0) {
				localSurvivors[BLOCK_SIZE - 1] = 0;
			}

			for (uint32 d = 1; d < BLOCK_SIZE; d <<= 1) {
				offset >>= 1;

				__syncthreads();

				if (blockId < d) {
					uint32 ai = offset * (2 * blockId + 1) - 1;
					uint32 bi = offset * (2 * blockId + 2) - 1;

					uint32 t = localSurvivors[ai];
					localSurvivors[ai] = localSurvivors[bi];
					localSurvivors[bi] += t;
				}
			}

			survivors[globalId].response = BAD_RESPONSE;

			__syncthreads();

			if (survived) {
				uint32 newThreadId = (blockIdx.y*blockDim.y) * PYRAMID.width + (blockIdx.x*blockDim.x) + localSurvivors[blockId];
				// save position and current response
				survivors[newThreadId].x = x;
				survivors[newThreadId].y = y;
				survivors[newThreadId].response = response;
			}
		}
	}

	__device__ void detectSurvivors(
		SurvivorData*	survivors,
		uint16			startStage,
		uint16			endStage)
	{
		__shared__ uint32 localSurvivors[BLOCK_SIZE];
		
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;
		const uint32 blockId = threadIdx.y * blockDim.x + threadIdx.x;
		const uint32 globalId = y * PYRAMID.width + x;

		localSurvivors[blockId] = 0;

		if (x < (PYRAMID.width - CLASSIFIER_WIDTH) && y < (PYRAMID.height - CLASSIFIER_HEIGHT))
		{
			float response = survivors[globalId].response;

			// kill the detection if its below threshold
			if (response < FINAL_THRESHOLD)
				return;
			
			const uint32 x = survivors[globalId].x;
			const uint32 y = survivors[globalId].y;

			__syncthreads();

			bool survived = eval(x, y, &response, startStage, endStage);

			localSurvivors[blockId] = static_cast<uint32>(survived);

			// up-sweep
			int offset = 1;
			for (uint32 d = BLOCK_SIZE >> 1; d > 0; d >>= 1, offset <<= 1) {
				__syncthreads();

				if (blockId < d) {
					uint32 ai = offset * (2 * blockId + 1) - 1;
					uint32 bi = offset * (2 * blockId + 2) - 1;
					localSurvivors[bi] += localSurvivors[ai];
				}
			}

			// down-sweep
			if (blockId == 0) {
				localSurvivors[BLOCK_SIZE - 1] = 0;
			}

			for (uint32 d = 1; d < BLOCK_SIZE; d <<= 1) {
				offset >>= 1;

				__syncthreads();

				if (blockId < d) {
					uint32 ai = offset * (2 * blockId + 1) - 1;
					uint32 bi = offset * (2 * blockId + 2) - 1;

					uint32 t = localSurvivors[ai];
					localSurvivors[ai] = localSurvivors[bi];
					localSurvivors[bi] += t;
				}
			}

			survivors[globalId].response = BAD_RESPONSE;

			__syncthreads();

			if (survived) {
				uint32 newThreadId = (blockIdx.y*blockDim.y) * PYRAMID.width + (blockIdx.x*blockDim.x) + localSurvivors[blockId];
				// save position and current response
				survivors[newThreadId].x = x;
				survivors[newThreadId].y = y;
				survivors[newThreadId].response = response;
			}
		}
	}

	__global__ void detectionKernel(
		Detection*		detections,
		uint32*			detectionCount,
		SurvivorData*	survivors)
	{
		detectSurvivorsInit(survivors, 16);
		detectSurvivors(survivors, 16, 32);
		detectSurvivors(survivors, 32, 64);
		detectSurvivors(survivors, 64, 128);
		detectSurvivors(survivors, 128, 256);
		detectSurvivors(survivors, 256, 512);
		detectDetections(survivors, detections, detectionCount, 512);
	}

	__device__ void sumRegions(float* values, uint32 x, uint32 y, Stage* stage)
	{
		values[0] = tex2D(textureImagePyramid, x, y);
		x += stage->width;
		values[1] = tex2D(textureImagePyramid, x, y);
		x += stage->width;
		values[2] = tex2D(textureImagePyramid, x, y);
		y += stage->height;
		values[5] = tex2D(textureImagePyramid, x, y);
		y += stage->height;
		values[8] = tex2D(textureImagePyramid, x, y);
		x -= stage->width;
		values[7] = tex2D(textureImagePyramid, x, y);
		x -= stage->width;
		values[6] = tex2D(textureImagePyramid, x, y);
		y -= stage->height;
		values[3] = tex2D(textureImagePyramid, x, y);
		x += stage->width;
		values[4] = tex2D(textureImagePyramid, x, y);
	}

	__device__ float evalLBP(uint32 x, uint32 y, Stage* stage)
	{
		const uint8 LBPOrder[8] = { 0, 1, 2, 5, 8, 7, 6, 3 };

		float values[9];

		sumRegions(values, x + (stage->width * 0.5f), y + (stage->height * 0.5f), stage);

		uint8 code = 0;
		for (uint8 i = 0; i < 8; ++i)
			code |= (values[LBPOrder[i]] > values[4]) << i;

		return tex1Dfetch(textureAlphas, stage->alphaOffset + code);
	}

	__device__ bool eval(uint32 x, uint32 y, float* response, uint16 startStage, uint16 endStage)
	{
		for (uint16 i = startStage; i < endStage; ++i) {
			Stage stage = stages[i];
			*response += evalLBP(x + stage.x, y + stage.y, &stage);
			if (*response < stage.thetaB) {
				return false;
			}
		}

		// final waldboost threshold
		return *response > FINAL_THRESHOLD;
	}


	__global__ void preprocessKernel(float* outData, uint8* inData)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < DEV_INFO.width && y < DEV_INFO.height)
		{
			uint32 pos = y * DEV_INFO.width + x;
			// convert to B&W
			outData[pos] = 0.299f * static_cast<float>(inData[3 * pos])
				+ 0.587f * static_cast<float>(inData[3 * pos + 1])
				+ 0.114f * static_cast<float>(inData[3 * pos + 2]);

			// clip to <0.0f;1.0f>
			outData[pos] /= 255.0f;
		}			
	}	

	__global__ void pyramidKernel(float* outData)
	{		
		// coords in the original image
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < PYRAMID.width && y < PYRAMID.height)
		{
			uint8 i = 1;
			for (; i < PYRAMID_IMAGE_COUNT; ++i)
			{
				if (y < PYRAMID.yOffsets[i])
					break;
			}
			uint8 imageIndex = i - 1;

			if (x < PYRAMID.imageWidths[imageIndex]) {
				float origX = static_cast<float>(x) / static_cast<float>(PYRAMID.imageWidths[imageIndex]) * static_cast<float>(DEV_INFO.width);
				float origY = static_cast<float>(y - PYRAMID.yOffsets[imageIndex]) / static_cast<float>(PYRAMID.imageHeights[imageIndex]) * static_cast<float>(DEV_INFO.height);

				float res = tex2D(textureWorkingImage, origX, origY);
				outData[y * PYRAMID.width + x] = res;
			}
		}
	}

	void WaldboostDetector::init(cv::Mat* image)
	{
		_info.width = image->cols;
		_info.height = image->rows;
		_info.imageSize = image->cols * image->rows;		
		_info.channels = image->channels();

		// pyramid image calcs
		float scale = _info.width / MAX_PYRAMID_WIDTH;
		float scaledHeight = _info.height / scale;
		float scaledWidth = _info.width / scale;
		float totalHeight = scaledHeight;

		_pyramid.yOffsets[0] = 0;		
		_pyramid.scales[0] = scale;		
		_pyramid.imageWidths[0] = scaledWidth;
		_pyramid.imageHeights[0] = scaledHeight;

		for (uint8 i = 1; i < 8; ++i)
		{
			scale *= SCALING_FACTOR;
			scaledHeight = _info.height / scale;
			scaledWidth = _info.width / scale;

			_pyramid.yOffsets[i] = totalHeight;
			_pyramid.scales[i] = scale;
			_pyramid.imageWidths[i] = scaledWidth;
			_pyramid.imageHeights[i] = scaledHeight;
		
			totalHeight += scaledHeight;
		}
		_pyramid.height = totalHeight;
		_pyramid.width = MAX_PYRAMID_WIDTH;
		_pyramid.imageSize = _pyramid.width * _pyramid.height;

		cudaMemcpyToSymbol(devPyramid, &_pyramid, sizeof(Pyramid));
		cudaMemcpyToSymbol(devInfo, &_info, sizeof(ImageInfo));
		cudaMemcpyToSymbol(stages, hostStages, sizeof(Stage) * STAGE_COUNT);

		cudaMalloc((void**)&_devOriginalImage, sizeof(uint8) * _info.imageSize * _info.channels);
		cudaMalloc((void**)&_devWorkingImage, sizeof(float) * _info.imageSize);
		cudaMalloc((void**)&_devPyramidImage, sizeof(float) * _pyramid.imageSize);
		cudaMemset((void**)&_devPyramidImage, 0, sizeof(float) * _pyramid.imageSize);
		cudaMalloc((void**)&_devDetections, sizeof(Detection) * MAX_DETECTIONS);
		cudaMalloc((void**)&_devDetectionCount, sizeof(uint32));
		cudaMemset((void**)&_devDetectionCount, 0, sizeof(uint32));
		cudaMalloc((void**)&_devSurvivors, sizeof(SurvivorData) * _pyramid.imageSize);
				
		cudaMalloc(&_devAlphaBuffer, STAGE_COUNT * ALPHA_COUNT * sizeof(float));
		cudaMemcpy(_devAlphaBuffer, alphas, STAGE_COUNT * ALPHA_COUNT * sizeof(float), cudaMemcpyHostToDevice);
		cudaChannelFormatDesc alphaDesc = cudaCreateChannelDesc<float>();
		cudaBindTexture(nullptr, &textureAlphas, _devAlphaBuffer, &alphaDesc, STAGE_COUNT * ALPHA_COUNT * sizeof(float));

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(nullptr, &textureWorkingImage, _devWorkingImage, &channelDesc, _info.width, _info.height, sizeof(float) * _info.width);

		cudaChannelFormatDesc pyramidDesc = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(nullptr, &textureImagePyramid, _devPyramidImage, &pyramidDesc, _pyramid.width, _pyramid.height, sizeof(float) * _pyramid.width);		

		#ifdef DEBUG
		_frameCount = 0;
		#endif
	}

	void WaldboostDetector::setAttributes(DetectorConfiguration const& config)
	{
		_config = config;
	}

	void WaldboostDetector::setImage(cv::Mat* image)
	{		
		#ifdef DEBUG
		std::cout << "------- FRAME " << _frameCount++ << "------" << std::endl;
		_totalTime = 0.f;
		#endif

		cudaMemcpy(_devOriginalImage, image->data, _info.imageSize * _info.channels * sizeof(uint8), cudaMemcpyHostToDevice);
		
		// allows max size 2048x1024
		dim3 grid(64, 32, 1);
		dim3 block(32, 32, 1);

		#ifdef DEBUG
		cudaEvent_t start_preprocess, stop_preprocess;
		float preprocess_time = 0.f;
		cudaEventCreate(&start_preprocess);
		cudaEventCreate(&stop_preprocess);				
		cudaEventRecord(start_preprocess);
		#endif

		preprocessKernel <<<grid, block>>>(_devWorkingImage, _devOriginalImage);

		#ifdef DEBUG
		cudaEventRecord(stop_preprocess);
		cudaEventSynchronize(stop_preprocess);
		cudaEventElapsedTime(&preprocess_time, start_preprocess, stop_preprocess);
		printf("Preprocessing kernel time: %f ms\n", preprocess_time);
		_totalTime += preprocess_time;
		#endif
								
		// should display black and white floating-point image
		#ifdef DEBUG
		cv::Mat tmp(cv::Size(_info.width, _info.height), CV_32FC1);
		cudaMemcpy(tmp.data, _devWorkingImage, _info.imageSize * sizeof(float), cudaMemcpyDeviceToHost);
		cv::imshow("converted to float and bw", tmp);
		cv::waitKey(WAIT_DELAY);
		#endif	

		// allows max size 2048x1024
		dim3 grid2(16, 32, 1);
		dim3 block2(32, 32, 1);

		#ifdef DEBUG
		cudaEvent_t start_pyramid, stop_pyramid;
		float pyramid_time = 0.f;
		cudaEventCreate(&start_pyramid);
		cudaEventCreate(&stop_pyramid);
		cudaEventRecord(start_pyramid);
		#endif

		pyramidKernel <<<grid2, block2>>>(_devPyramidImage);

		#ifdef DEBUG
		cudaEventRecord(stop_pyramid);
		cudaEventSynchronize(stop_pyramid);
		cudaEventElapsedTime(&pyramid_time, start_pyramid, stop_pyramid);
		printf("Pyramid kernel time: %f ms\n", pyramid_time);
		_totalTime += pyramid_time;
		#endif

		// should display black and white floating-point image
		#ifdef DEBUG
		cv::Mat tmp2(cv::Size(_pyramid.width, _pyramid.height), CV_32FC1);
		cudaMemcpy(tmp2.data, _devPyramidImage, _pyramid.imageSize * sizeof(float), cudaMemcpyDeviceToHost);
		cv::imshow("pyramid image", tmp2);
		cv::waitKey(WAIT_DELAY);
		#endif	
	}

	void WaldboostDetector::run()
	{
		// allows max size 2048x1024
		dim3 grid(16, 32, 1);
		dim3 block(32, 32, 1);

		cudaMemset(_devDetectionCount, 0, sizeof(uint32));
		
		#ifdef DEBUG
		cudaEvent_t start_detection, stop_detection;
		float detection_time = 0.f;
		cudaEventCreate(&start_detection);
		cudaEventCreate(&stop_detection);
		cudaEventRecord(start_detection);
		#endif

		detectionKernel <<<grid, block>>>(_devDetections, _devDetectionCount, _devSurvivors);
		
		#ifdef DEBUG
		cudaEventRecord(stop_detection);
		cudaEventSynchronize(stop_detection);
		cudaEventElapsedTime(&detection_time, start_detection, stop_detection);
		printf("Detection kernel time: %f ms\n", detection_time);
		_totalTime += detection_time;
		printf("TOTAL KERNEL TIME: %f ms\n", _totalTime);
		#endif

		cv::Mat tmp(cv::Size(_pyramid.width, _pyramid.height), CV_32FC1);
		uint32 detectionCount;
		cudaMemcpy(&detectionCount, _devDetectionCount, sizeof(uint32), cudaMemcpyDeviceToHost);

		#ifdef DEBUG
		std::cout << "DETECTION COUNT: " << detectionCount << std::endl;		
		#endif

		Detection detections[MAX_DETECTIONS];
		cudaMemcpy(&detections, _devDetections, detectionCount * sizeof(Detection), cudaMemcpyDeviceToHost);
		cudaMemcpy(tmp.data, _devPyramidImage, _pyramid.imageSize * sizeof(float), cudaMemcpyDeviceToHost);

		#ifdef DEBUG
		for (uint32 i = 0; i < detectionCount; ++i)
			cv::rectangle(tmp, cvPoint(detections[i].x, detections[i].y), cvPoint(detections[i].x + detections[i].width, detections[i].y + detections[i].height), CV_RGB(255, 255, 255), 1);

		cv::imshow("detections", tmp);
		cv::waitKey(WAIT_DELAY);
		#endif
	}

	void WaldboostDetector::free()
	{
		cudaUnbindTexture(textureWorkingImage);
		cudaUnbindTexture(textureAlphas);
		cudaUnbindTexture(textureImagePyramid);

		cudaFree(_devOriginalImage);
		cudaFree(_devWorkingImage);
		cudaFree(_devPyramidImage);
		cudaFree(_devAlphaBuffer);
		cudaFree(_devDetections);
		cudaFree(_devDetectionCount);
		cudaFree(_devSurvivors);
	}
}
