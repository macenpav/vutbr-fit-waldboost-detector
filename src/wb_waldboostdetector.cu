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

		if ((x < (PYRAMID.width - CLASSIFIER_WIDTH) && y < (PYRAMID.height - CLASSIFIER_HEIGHT)) ||
			(x < (PYRAMID.widthL1 - CLASSIFIER_WIDTH) && y < (PYRAMID.heightL1 / 2 - CLASSIFIER_HEIGHT)))
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

		if ((x < (PYRAMID.width - CLASSIFIER_WIDTH) && y < (PYRAMID.height - CLASSIFIER_HEIGHT)) ||
			(x < (PYRAMID.widthL1 - CLASSIFIER_WIDTH) && y < (PYRAMID.heightL1 / 2 - CLASSIFIER_HEIGHT)))
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

		if ((x < (PYRAMID.width - CLASSIFIER_WIDTH) && y < (PYRAMID.height - CLASSIFIER_HEIGHT)) ||
			(x < (PYRAMID.widthL1 - CLASSIFIER_WIDTH) && y < (PYRAMID.heightL1 / 2 - CLASSIFIER_HEIGHT)))
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
		detectSurvivorsInit(survivors, 1);
		detectSurvivors(survivors, 1, 8);
		detectSurvivors(survivors, 8, 64);
		detectSurvivors(survivors, 64, 256);
		detectSurvivors(survivors, 256, 512);
		detectDetections(survivors, detections, detectionCount, 512);
	}

	__device__ void sumRegions(float* values, float x, float y, Stage* stage)
	{
		values[0] = tex2D(textureImagePyramid1, x, y);
		x += stage->width;
		values[1] = tex2D(textureImagePyramid1, x, y);
		x += stage->width;
		values[2] = tex2D(textureImagePyramid1, x, y);
		y += stage->height;
		values[5] = tex2D(textureImagePyramid1, x, y);
		y += stage->height;
		values[8] = tex2D(textureImagePyramid1, x, y);
		x -= stage->width;
		values[7] = tex2D(textureImagePyramid1, x, y);
		x -= stage->width;
		values[6] = tex2D(textureImagePyramid1, x, y);
		y -= stage->height;
		values[3] = tex2D(textureImagePyramid1, x, y);
		x += stage->width;
		values[4] = tex2D(textureImagePyramid1, x, y);
	}

	__device__ float evalLBP(uint32 x, uint32 y, Stage* stage)
	{
		const uint8 LBPOrder[8] = { 0, 1, 2, 5, 8, 7, 6, 3 };

		float values[9];

		sumRegions(values, static_cast<float>(x) + (static_cast<float>(stage->width) * 0.5f), y + (static_cast<float>(stage->height) * 0.5f), stage);

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

	__global__ void pyramidKernelL0(float* outData)
	{		
		// coords in the original image
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < PYRAMID.width && y < PYRAMID.height)
		{
			uint32 offsetX = 0;
			
			Octave oct = PYRAMID.octaves[0];
			float res;
			uint8 i = 1;
			for (; i < WB_LEVELS_PER_OCTAVE; ++i)
			{
				if (y < oct.images[i].offsetY)
					break;
			}

			if (x > oct.images[i - 1].width)
			{
				res = 0.f;
			}
			else
			{
				float origX = static_cast<float>(x - oct.images[i - 1].offsetX) / static_cast<float>(oct.images[i - 1].width) * static_cast<float>(DEV_INFO.width);
				float origY = static_cast<float>(y - oct.images[i - 1].offsetY) / static_cast<float>(oct.images[i - 1].height) * static_cast<float>(DEV_INFO.height);
				res = tex2D(textureWorkingImage, origX, origY);
			}

			outData[y * PYRAMID.width + x] = res;			
		}		
	}

	__global__ void pyramidKernelL1(float* outData)
	{
		// coords in the original image
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;
				
		if (x < PYRAMID.width && y < PYRAMID.height)
		{
			float res = tex2D(textureImagePyramid0, x + 0.5f, y + 0.5f);
			outData[y * PYRAMID.widthL1 + x] = res;
			return;
		}

		
		if (x < PYRAMID.widthL1 && y < PYRAMID.heightL1 / 2)
		{
			float origX = static_cast<float>(x - PYRAMID.width) * 2.f;
			float origY = static_cast<float>(y)* 2.f;
			float res = tex2D(textureImagePyramid0, origX, origY);
			outData[(y * PYRAMID.widthL1) + x] = res;
			return;
		}		
					;
	}

	void WaldboostDetector::_precalcPyramid()
	{		
		float scaledWidth, scaledHeight;
		float parentWidth = _info.width, parentHeight = _info.height;
		uint32 currentOffset = 0, parentOffset = 0, currentOffsetX = 0, currentOffsetY = 0;
		const uint32 pitch = _info.width;

		// 1st octave
		uint8 oct = 0;
		
		float scale = 1.0f;
		for (uint8 lvl = 0; lvl < WB_LEVELS_PER_OCTAVE; ++lvl)
		{
			ImageData data;

			scaledHeight = parentHeight / scale;
			scaledWidth = parentWidth / scale;

			data.width = static_cast<uint32>(scaledWidth);
			data.height = static_cast<uint32>(scaledHeight);				
			data.offsetY = currentOffsetY;
			data.offsetX = currentOffsetX;

			scale *= WB_SCALING_FACTOR;
			currentOffsetY += data.height;

			_pyramid.octaves[oct].images[lvl] = data;
		}

		currentOffsetX += _pyramid.octaves[oct].images[0].width;
		parentWidth = _pyramid.octaves[oct].images[0].width;
		parentHeight = _pyramid.octaves[oct].images[0].height;
		
		_pyramid.height = currentOffsetY;
		_pyramid.heightL1 = _pyramid.height;
		_pyramid.width = currentOffsetX;
		_pyramid.widthL1 = _pyramid.width + _pyramid.width / 2;
		_pyramid.imageSize = _pyramid.width * _pyramid.height;	
		_pyramid.imageSizeL1 = (_pyramid.width + _pyramid.width / 2) * _pyramid.height;

		/*Octave oct = _pyramid.octaves[0];
		for (uint32 i = 0; i < WB_LEVELS_PER_OCTAVE; ++i) {
			std::cout << i << ". image " << oct.images[i].width << "x" << oct.images[i].height << std::endl;
			std::cout << i << ". image offsets x: " << oct.images[i].offsetX<< " y: " << oct.images[i].offsetY << std::endl;
		}

		std::cout << "pyramid size: " << _pyramid.width << "x" << _pyramid.height << std::endl;	*/
	}

	void WaldboostDetector::init(cv::Mat* image)
	{
		_info.width = image->cols;
		_info.height = image->rows;
		_info.imageSize = image->cols * image->rows;		
		_info.channels = image->channels();		

		_precalcPyramid();

		cudaMemcpyToSymbol(devPyramid, &_pyramid, sizeof(Pyramid));
		cudaMemcpyToSymbol(devInfo, &_info, sizeof(ImageInfo));
		cudaMemcpyToSymbol(stages, hostStages, sizeof(Stage) * STAGE_COUNT);

		cudaMalloc((void**)&_devOriginalImage, sizeof(uint8) * _info.imageSize * _info.channels);
		cudaMalloc((void**)&_devWorkingImage, sizeof(float) * _info.imageSize);
		cudaMalloc((void**)&_devPyramidImage0, sizeof(float) * _pyramid.imageSize);		
		cudaMalloc((void**)&_devPyramidImage1, sizeof(float) * _pyramid.imageSizeL1);
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

		 
		cudaChannelFormatDesc pyramidDesc0 = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(nullptr, &textureImagePyramid0, _devPyramidImage0, &pyramidDesc0, _pyramid.width, _pyramid.height, sizeof(float) * _pyramid.width);		
		
		cudaChannelFormatDesc pyramidDesc1 = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(nullptr, &textureImagePyramid1, _devPyramidImage1, &pyramidDesc1, _pyramid.widthL1, _pyramid.heightL1, sizeof(float) * (_pyramid.widthL1));

		#ifdef WB_DEBUG
		_frameCount = 0;
		#endif
	}

	void WaldboostDetector::setAttributes(DetectorConfiguration const& config)
	{
		_config = config;
	}

	void WaldboostDetector::setImage(cv::Mat* image)
	{		
		textureWorkingImage.addressMode[0] = cudaAddressModeClamp;
		textureWorkingImage.addressMode[1] = cudaAddressModeClamp;
		textureWorkingImage.filterMode = cudaFilterModeLinear;
		textureWorkingImage.normalized = false;

		textureImagePyramid0.addressMode[0] = cudaAddressModeClamp;
		textureImagePyramid0.addressMode[1] = cudaAddressModeClamp;
		textureImagePyramid0.filterMode = cudaFilterModeLinear;
		textureImagePyramid0.normalized = false;

		textureImagePyramid1.addressMode[0] = cudaAddressModeClamp;
		textureImagePyramid1.addressMode[1] = cudaAddressModeClamp;
		textureImagePyramid1.filterMode = cudaFilterModeLinear;
		textureImagePyramid1.normalized = false;

		#ifdef WB_DEBUG
		std::cout << "------- FRAME " << _frameCount++ << "------" << std::endl;
		_totalTime = 0.f;
		#endif

		cudaMemcpy(_devOriginalImage, image->data, _info.imageSize * _info.channels * sizeof(uint8), cudaMemcpyHostToDevice);
		
		// allows max size 2048x1024
		dim3 grid(64, 32, 1);
		dim3 block(32, 32, 1);

		#ifdef WB_DEBUG
		cudaEvent_t start_preprocess, stop_preprocess;
		float preprocess_time = 0.f;
		cudaEventCreate(&start_preprocess);
		cudaEventCreate(&stop_preprocess);				
		cudaEventRecord(start_preprocess);
		#endif

		preprocessKernel <<<grid, block>>>(_devWorkingImage, _devOriginalImage);

		#ifdef WB_DEBUG
		cudaEventRecord(stop_preprocess);
		cudaEventSynchronize(stop_preprocess);
		cudaEventElapsedTime(&preprocess_time, start_preprocess, stop_preprocess);
		printf("Preprocessing kernel time: %f ms\n", preprocess_time);
		_totalTime += preprocess_time;
		#endif
								
		// should display black and white floating-point image
		#ifdef WB_DEBUG
		cv::Mat tmp(cv::Size(_info.width, _info.height), CV_32FC1);
		cudaMemcpy(tmp.data, _devWorkingImage, _info.imageSize * sizeof(float), cudaMemcpyDeviceToHost);
		cv::imshow("converted to float and bw", tmp);
		cv::waitKey(WAIT_DELAY);
		#endif	

		// allows max size 2048x1024
		dim3 grid2(64, 64, 1);
		dim3 block2(32, 32, 1);

		#ifdef WB_DEBUG
		cudaEvent_t start_pyramid, stop_pyramid;
		float pyramid_time = 0.f;
		cudaEventCreate(&start_pyramid);
		cudaEventCreate(&stop_pyramid);
		cudaEventRecord(start_pyramid);
		#endif

		pyramidKernelL0 <<<grid2, block2>>>(_devPyramidImage0);
		pyramidKernelL1 <<<grid2, block2>>>(_devPyramidImage1);

		#ifdef WB_DEBUG
		cudaEventRecord(stop_pyramid);
		cudaEventSynchronize(stop_pyramid);
		cudaEventElapsedTime(&pyramid_time, start_pyramid, stop_pyramid);
		printf("Pyramid kernel time: %f ms\n", pyramid_time);
		_totalTime += pyramid_time;
		#endif

		// should display black and white floating-point image
		#ifdef WB_DEBUG
		cv::Mat tmp2(cv::Size(_pyramid.widthL1, _pyramid.heightL1), CV_32FC1);
		cudaMemcpy(tmp2.data, _devPyramidImage1, _pyramid.imageSizeL1 * sizeof(float), cudaMemcpyDeviceToHost);
		cv::imshow("pyramid image", tmp2);
		cv::waitKey(WAIT_DELAY);
		#endif	
	}

	void WaldboostDetector::run()
	{
		// allows max size 2048x1024
		dim3 grid(64, 64, 1);
		dim3 block(32, 32, 1);

		cudaMemset(_devDetectionCount, 0, sizeof(uint32));
		
		#ifdef WB_DEBUG
		cudaEvent_t start_detection, stop_detection;
		float detection_time = 0.f;
		cudaEventCreate(&start_detection);
		cudaEventCreate(&stop_detection);
		cudaEventRecord(start_detection);
		#endif

		detectionKernel <<<grid, block>>>(_devDetections, _devDetectionCount, _devSurvivors);
		
		#ifdef WB_DEBUG
		cudaEventRecord(stop_detection);
		cudaEventSynchronize(stop_detection);
		cudaEventElapsedTime(&detection_time, start_detection, stop_detection);
		printf("Detection kernel time: %f ms\n", detection_time);
		_totalTime += detection_time;
		printf("TOTAL KERNEL TIME: %f ms\n", _totalTime);
		#endif

		cv::Mat tmp(cv::Size(_pyramid.widthL1, _pyramid.heightL1), CV_32FC1);
		uint32 detectionCount;
		cudaMemcpy(&detectionCount, _devDetectionCount, sizeof(uint32), cudaMemcpyDeviceToHost);

		#ifdef WB_DEBUG
		std::cout << "DETECTION COUNT: " << detectionCount << std::endl;		
		#endif

		Detection detections[MAX_DETECTIONS];
		cudaMemcpy(&detections, _devDetections, detectionCount * sizeof(Detection), cudaMemcpyDeviceToHost);
		cudaMemcpy(tmp.data, _devPyramidImage1, _pyramid.imageSizeL1 * sizeof(float), cudaMemcpyDeviceToHost);

		#ifdef WB_DEBUG
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
		cudaUnbindTexture(textureImagePyramid0);
		cudaUnbindTexture(textureImagePyramid1);

		cudaFree(_devOriginalImage);
		cudaFree(_devWorkingImage);
		cudaFree(_devPyramidImage0);
		cudaFree(_devPyramidImage1);
		cudaFree(_devAlphaBuffer);
		cudaFree(_devDetections);
		cudaFree(_devDetectionCount);
		cudaFree(_devSurvivors);
	}
}
