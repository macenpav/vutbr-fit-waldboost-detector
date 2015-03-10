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
		cudaTextureObject_t inTexture,
		Detection*		detections,
		uint32*			detectionCount,
		uint16			startStage)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;		
		const uint32 globalId = y * PYRAMID.canvasWidth[WB_MAX_OCTAVE_INDEX] + x;

		if (x < (PYRAMID.canvasWidth[WB_MAX_OCTAVE_INDEX] - CLASSIFIER_WIDTH) && y < (PYRAMID.canvasHeight[WB_MAX_OCTAVE_INDEX] - CLASSIFIER_HEIGHT))
		{
			float response = survivors[globalId].response;

			if (response < FINAL_THRESHOLD)
				return;

			const uint32 x = survivors[globalId].x;
			const uint32 y = survivors[globalId].y;

			__syncthreads();

			bool survived = eval(inTexture, x, y, &response, startStage, STAGE_COUNT);
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
		cudaTextureObject_t inTexture,
		uint16			endStage)
	{
		__shared__ uint32 localSurvivors[BLOCK_SIZE];

		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;
		const uint32 blockId = threadIdx.y * blockDim.x + threadIdx.x;
		const uint32 globalId = y * PYRAMID.canvasWidth[WB_MAX_OCTAVE_INDEX] + x;

		localSurvivors[blockId] = 0;

		if (x < (PYRAMID.canvasWidth[WB_MAX_OCTAVE_INDEX] - CLASSIFIER_WIDTH) && y < (PYRAMID.canvasHeight[WB_MAX_OCTAVE_INDEX] - CLASSIFIER_HEIGHT))
		{
			float response = 0.0f;
			bool survived = eval(inTexture, x, y, &response, 0, endStage);

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
				uint32 newThreadId = (blockIdx.y*blockDim.y) * PYRAMID.canvasWidth[WB_MAX_OCTAVE_INDEX] + (blockIdx.x*blockDim.x) + localSurvivors[blockId];
				// save position and current response
				survivors[newThreadId].x = x;
				survivors[newThreadId].y = y;
				survivors[newThreadId].response = response;
			}
		}
	}

	__device__ void detectSurvivors(
		SurvivorData*	survivors,
		cudaTextureObject_t inTexture,
		uint16			startStage,
		uint16			endStage)
	{
		__shared__ uint32 localSurvivors[BLOCK_SIZE];
		
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;
		const uint32 blockId = threadIdx.y * blockDim.x + threadIdx.x;
		const uint32 globalId = y * PYRAMID.canvasWidth[WB_MAX_OCTAVE_INDEX] + x;

		localSurvivors[blockId] = 0;

		if (x < (PYRAMID.canvasWidth[WB_MAX_OCTAVE_INDEX] - CLASSIFIER_WIDTH) && y < (PYRAMID.canvasHeight[WB_MAX_OCTAVE_INDEX] - CLASSIFIER_HEIGHT))
		{
			float response = survivors[globalId].response;

			// kill the detection if its below threshold
			if (response < FINAL_THRESHOLD)
				return;
			
			const uint32 x = survivors[globalId].x;
			const uint32 y = survivors[globalId].y;

			__syncthreads();

			bool survived = eval(inTexture, x, y, &response, startStage, endStage);

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
				uint32 newThreadId = (blockIdx.y*blockDim.y) * PYRAMID.canvasWidth[WB_MAX_OCTAVE_INDEX] + (blockIdx.x*blockDim.x) + localSurvivors[blockId];
				// save position and current response
				survivors[newThreadId].x = x;
				survivors[newThreadId].y = y;
				survivors[newThreadId].response = response;
			}
		}
	}

	__global__ void detectionKernel(
		Detection*			detections,
		uint32*				detectionCount,
		SurvivorData*		survivors,
		cudaTextureObject_t inTexture)
	{
		detectSurvivorsInit(survivors, inTexture, 1);
		detectSurvivors(survivors, inTexture, 1, 8);
		detectSurvivors(survivors, inTexture, 8, 64);
		detectSurvivors(survivors, inTexture, 64, 256);
		detectSurvivors(survivors, inTexture, 256, 512);
		detectDetections(survivors, inTexture, detections, detectionCount, 512);
	}

	__device__ void sumRegions(cudaTextureObject_t inTexture, float* values, float x, float y, Stage* stage)
	{
		values[0] = tex2D<float>(inTexture, x, y);
		x += stage->width;
		values[1] = tex2D<float>(inTexture, x, y);
		x += stage->width;
		values[2] = tex2D<float>(inTexture, x, y);
		y += stage->height;
		values[5] = tex2D<float>(inTexture, x, y);
		y += stage->height;
		values[8] = tex2D<float>(inTexture, x, y);
		x -= stage->width;
		values[7] = tex2D<float>(inTexture, x, y);
		x -= stage->width;
		values[6] = tex2D<float>(inTexture, x, y);
		y -= stage->height;
		values[3] = tex2D<float>(inTexture, x, y);
		x += stage->width;
		values[4] = tex2D<float>(inTexture, x, y);
	}

	__device__ float evalLBP(cudaTextureObject_t inTexture, uint32 x, uint32 y, Stage* stage)
	{
		const uint8 LBPOrder[8] = { 0, 1, 2, 5, 8, 7, 6, 3 };

		float values[9];

		sumRegions(inTexture, values, static_cast<float>(x) + (static_cast<float>(stage->width) * 0.5f), y + (static_cast<float>(stage->height) * 0.5f), stage);

		uint8 code = 0;
		for (uint8 i = 0; i < 8; ++i)
			code |= (values[LBPOrder[i]] > values[4]) << i;

		return tex1Dfetch(textureAlphas, stage->alphaOffset + code);
	}

	__device__ bool eval(cudaTextureObject_t inTexture, uint32 x, uint32 y, float* response, uint16 startStage, uint16 endStage)
	{
		for (uint16 i = startStage; i < endStage; ++i) {
			Stage stage = stages[i];
			*response += evalLBP(inTexture, x + stage.x, y + stage.y, &stage);
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

		if (x < PYRAMID.width[0] && y < PYRAMID.height[0])
		{			
			Octave oct = PYRAMID.octaves[0];
			float res;
			uint8 i = 1;
			for (; i < WB_LEVELS_PER_OCTAVE; ++i)
			{
				if (y < oct.images[i].offsetY)
					break;
			}

			if (x > oct.images[i - 1].width)			
				res = 0.f;			
			else
			{
				float origX = static_cast<float>(x - oct.images[i - 1].offsetX) / static_cast<float>(oct.images[i - 1].width) * static_cast<float>(DEV_INFO.width);
				float origY = static_cast<float>(y - oct.images[i - 1].offsetY) / static_cast<float>(oct.images[i - 1].height) * static_cast<float>(DEV_INFO.height);
				res = tex2D(textureWorkingImage, origX, origY);
			}

			outData[y * PYRAMID.width[0] + x] = res;			
		}		
	}

	/*__global__ void pyramidKernelL1(float* outData)
	{
		// coords in the original image
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;				

		if (x < PYRAMID.widthL1 && y < PYRAMID.heightL1)
			outData[(y * PYRAMID.widthL1) + x] = 0.f;

		__syncthreads();

		if (x < PYRAMID.width && y < PYRAMID.height)
		{			
			outData[y * PYRAMID.widthL1 + x] = tex2D(textureImagePyramid0, x + 0.5f, y + 0.5f);
			return;
		}		
		
		if (x < PYRAMID.widthL1 && y < PYRAMID.heightL1 / 2)
		{
			float origX = static_cast<float>(x - PYRAMID.width) * 2.f;
			float origY = static_cast<float>(y)* 2.f;			
			outData[(y * PYRAMID.widthL1) + x] = tex2D(textureImagePyramid0, origX, origY);
			return;
		}
	}*/

	__global__ void pyramidKernelUni(float* outData, cudaTextureObject_t inData, uint8 level)
	{
		// coords in the original image
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		const uint32 canvasWidth = PYRAMID.canvasWidth[level];
		const uint32 canvasHeight = PYRAMID.canvasHeight[level];

		// set the image to black
		if (x < canvasWidth && y < canvasHeight)
		{ 
			outData[y * canvasWidth + x] = 0.f;

			__syncthreads();

			const uint32 width = PYRAMID.width[level];
			const uint32 height = PYRAMID.height[level];	

			const uint32 prevCanvasWidth = PYRAMID.canvasWidth[level - 1];
			const uint32 prevWidth = PYRAMID.width[level - 1];

			// copy previously generated pyramids
			if (x < canvasWidth - width && y < canvasHeight)
			{
				outData[y * canvasWidth + x] = tex2D<float>(inData, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f); // +0.5f takes center of the texel - no interpolation
				return;
			}

			// generate a new pyramid level by downsampling the previous one 2x
			if (x < canvasWidth && y < height)
			{							
				float prevX = static_cast<float>(x - prevCanvasWidth) * 2.f + static_cast<float>(prevCanvasWidth - prevWidth);
				float prevY = static_cast<float>(y) * 2.f;			

				outData[y * canvasWidth + x] = tex2D<float>(inData, prevX, prevY);
				return;
			}
		}
	}

	void WaldboostDetector::_precalcPyramid()
	{		
		float scaledWidth, scaledHeight;		
		uint32 currentOffsetX = 0, currentOffsetY = 0;
		
		uint32 totalWidth = 0;
		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
		{		
			_pyramid.width[oct] = _info.width >> oct;			
			totalWidth += _pyramid.width[oct];
			_pyramid.canvasWidth[oct] = totalWidth;			

			float scale = pow(2.f, oct);
			for (uint8 lvl = 0; lvl < WB_LEVELS_PER_OCTAVE; ++lvl)
			{
				ImageData data;

				scaledWidth = _info.width / scale;
				scaledHeight = _info.height / scale;

				data.width = static_cast<uint32>(scaledWidth);
				data.height = static_cast<uint32>(scaledHeight);
				data.offsetY = currentOffsetY;
				data.offsetX = currentOffsetX;

				scale *= WB_SCALING_FACTOR;
				currentOffsetY += data.height;

				_pyramid.octaves[oct].images[lvl] = data;
			}

			_pyramid.canvasHeight[oct] = (oct == 0) ? currentOffsetY : _pyramid.canvasHeight[0];
			_pyramid.height[oct] = _pyramid.canvasHeight[oct] >> oct;
			_pyramid.canvasImageSize[oct] = _pyramid.height[oct] * _pyramid.width[oct];
			_pyramid.canvasImageSize[oct] = _pyramid.canvasWidth[oct] * _pyramid.canvasHeight[oct];
			currentOffsetX += _pyramid.canvasWidth[oct];
		}

		Octave oct = _pyramid.octaves[0];
		for (uint32 i = 0; i < WB_LEVELS_PER_OCTAVE; ++i) {
			std::cout << i << ". image " << oct.images[i].width << "x" << oct.images[i].height << std::endl;
			std::cout << i << ". image offsets x: " << oct.images[i].offsetX<< " y: " << oct.images[i].offsetY << std::endl;
		}

		for (uint32 i = 0; i < WB_OCTAVES; ++i) {
			std::cout << i << " canvas pyramid size: " << _pyramid.canvasWidth[i] << "x" << _pyramid.canvasHeight[i] << std::endl;
		}
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
		
		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
			cudaMalloc((void**)&_devPyramidImage[oct], sizeof(float) * _pyramid.canvasImageSize[oct]);

		cudaMalloc((void**)&_devDetections, sizeof(Detection) * MAX_DETECTIONS);
		cudaMalloc((void**)&_devDetectionCount, sizeof(uint32));
		cudaMemset((void**)&_devDetectionCount, 0, sizeof(uint32));
		cudaMalloc((void**)&_devSurvivors, sizeof(SurvivorData) * _pyramid.canvasImageSize[WB_MAX_OCTAVE_INDEX]);
				
		cudaMalloc(&_devAlphaBuffer, STAGE_COUNT * ALPHA_COUNT * sizeof(float));
		cudaMemcpy(_devAlphaBuffer, alphas, STAGE_COUNT * ALPHA_COUNT * sizeof(float), cudaMemcpyHostToDevice);
		cudaChannelFormatDesc alphaDesc = cudaCreateChannelDesc<float>();
		cudaBindTexture(nullptr, &textureAlphas, _devAlphaBuffer, &alphaDesc, STAGE_COUNT * ALPHA_COUNT * sizeof(float));

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(nullptr, &textureWorkingImage, _devWorkingImage, &channelDesc, _info.width, _info.height, sizeof(float) * _info.width);

		#ifdef WB_DEBUG
		_frameCount = 0;
		#endif
	}

	void WaldboostDetector::setAttributes(DetectorConfiguration const& config)
	{
		_config = config;
	}

	__global__ void copyKernel(float* out, cudaTextureObject_t in, uint32 width, uint32 height)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < width && y < height)
		{
			out[y * width + x] = tex2D<float>(in, x + 0.5f, y + 0.5f);
		}
	}

	void WaldboostDetector::setImage(cv::Mat* image)
	{		
		textureWorkingImage.addressMode[0] = cudaAddressModeClamp;
		textureWorkingImage.addressMode[1] = cudaAddressModeClamp;
		textureWorkingImage.filterMode = cudaFilterModeLinear;
		textureWorkingImage.normalized = false;

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

		pyramidKernelL0 <<<grid2, block2>>>(_devPyramidImage[0]);	
		
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = _devPyramidImage[0];
		resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
		resDesc.res.pitch2D.desc.x = 32; // bits per channel
		resDesc.res.pitch2D.height = _pyramid.height[0];
		resDesc.res.pitch2D.width = _pyramid.width[0];
		resDesc.res.pitch2D.pitchInBytes = _pyramid.width[0] * sizeof(float);

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;
		
		cudaCreateTextureObject(&_texturePyramid[0], &resDesc, &texDesc, NULL);

#ifdef WB_DEBUG
		float* pyramidCopiedFromTxt;
		cudaMalloc((void**)&pyramidCopiedFromTxt, _pyramid.canvasImageSize[0] * sizeof(float));
		copyKernel <<<grid2, block2>>>(pyramidCopiedFromTxt, _texturePyramid[0], _pyramid.canvasWidth[0], _pyramid.canvasHeight[0]);
		
		cv::Mat mat(cv::Size(_pyramid.width[0], _pyramid.height[0]), CV_32FC1);
		cudaMemcpy(mat.data, pyramidCopiedFromTxt, _pyramid.canvasImageSize[0] * sizeof(float), cudaMemcpyDeviceToHost);
		cv::imshow("pyramid image from texture", mat);
		cv::waitKey(WAIT_DELAY);

		cv::Mat mat2(cv::Size(_pyramid.width[0], _pyramid.height[0]), CV_32FC1);
		cudaMemcpy(mat2.data, _devPyramidImage[0], _pyramid.canvasImageSize[0] * sizeof(float), cudaMemcpyDeviceToHost);
		cv::imshow("pyramid image from array", mat2);
		cv::waitKey(WAIT_DELAY);
		
		cudaFree(pyramidCopiedFromTxt);		
#endif
	

		dim3 grid3(128, 64, 1);
		dim3 block3(32, 32, 1);
		for (uint8 oct = 1; oct < WB_OCTAVES; ++oct)
		{
			pyramidKernelUni<<<grid3, block3>>>(_devPyramidImage[oct], _texturePyramid[oct-1], oct);			

			cudaResourceDesc resDescUni;
			memset(&resDescUni, 0, sizeof(resDescUni));
			resDescUni.resType = cudaResourceTypePitch2D;
			resDescUni.res.pitch2D.devPtr = _devPyramidImage[oct];
			resDescUni.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
			resDescUni.res.pitch2D.desc.x = 32; // bits per channel
			resDescUni.res.pitch2D.height = _pyramid.canvasHeight[oct];
			resDescUni.res.pitch2D.width = _pyramid.canvasWidth[oct];
			resDescUni.res.pitch2D.pitchInBytes = _pyramid.canvasWidth[oct] * sizeof(float);

			cudaTextureDesc texDescUni;
			memset(&texDescUni, 0, sizeof(texDescUni));
			texDescUni.readMode = cudaReadModeElementType;
			texDescUni.filterMode = cudaFilterModeLinear;
			texDescUni.addressMode[0] = cudaAddressModeClamp;
			texDescUni.addressMode[1] = cudaAddressModeClamp;
			texDescUni.normalizedCoords = false;

			cudaCreateTextureObject(&_texturePyramid[oct], &resDescUni, &texDescUni, NULL);
		}
				

		#ifdef WB_DEBUG
		cudaEventRecord(stop_pyramid);
		cudaEventSynchronize(stop_pyramid);
		cudaEventElapsedTime(&pyramid_time, start_pyramid, stop_pyramid);
		printf("Pyramid kernel time: %f ms\n", pyramid_time);
		_totalTime += pyramid_time;
		#endif

		// should display black and white floating-point image
		#ifdef WB_DEBUG
		float* copied;
		cudaMalloc((void**)&copied, _pyramid.canvasImageSize[WB_MAX_OCTAVE_INDEX] * sizeof(float));
		copyKernel <<<grid3, block3>>>(copied, _texturePyramid[WB_MAX_OCTAVE_INDEX], _pyramid.canvasWidth[WB_MAX_OCTAVE_INDEX], _pyramid.canvasHeight[WB_MAX_OCTAVE_INDEX]);
	

		cv::Mat mat3(cv::Size(_pyramid.canvasWidth[WB_MAX_OCTAVE_INDEX], _pyramid.canvasHeight[WB_MAX_OCTAVE_INDEX]), CV_32FC1);
		cudaMemcpy(mat3.data, copied, _pyramid.canvasImageSize[WB_MAX_OCTAVE_INDEX] * sizeof(float), cudaMemcpyDeviceToHost);
		cv::imshow("aa", mat3);
		cv::waitKey(WAIT_DELAY);
		cudaFree(copied);
		#endif
	}

	void WaldboostDetector::run()
	{
		// allows max size 2048x1024
		dim3 grid(128, 64, 1);
		dim3 block(32, 32, 1);

		cudaMemset(_devDetectionCount, 0, sizeof(uint32));
		
		#ifdef WB_DEBUG
		cudaEvent_t start_detection, stop_detection;
		float detection_time = 0.f;
		cudaEventCreate(&start_detection);
		cudaEventCreate(&stop_detection);
		cudaEventRecord(start_detection);
		#endif

		detectionKernel <<<grid, block>>>(_devDetections, _devDetectionCount, _devSurvivors, _texturePyramid[WB_MAX_OCTAVE_INDEX]);
		
		#ifdef WB_DEBUG
		cudaEventRecord(stop_detection);
		cudaEventSynchronize(stop_detection);
		cudaEventElapsedTime(&detection_time, start_detection, stop_detection);
		printf("Detection kernel time: %f ms\n", detection_time);
		_totalTime += detection_time;
		printf("TOTAL KERNEL TIME: %f ms\n", _totalTime);
		#endif

		cv::Mat tmp(cv::Size(_pyramid.canvasWidth[WB_MAX_OCTAVE_INDEX], _pyramid.canvasHeight[WB_MAX_OCTAVE_INDEX]), CV_32FC1);
		uint32 detectionCount;
		cudaMemcpy(&detectionCount, _devDetectionCount, sizeof(uint32), cudaMemcpyDeviceToHost);

		#ifdef WB_DEBUG
		std::cout << "DETECTION COUNT: " << detectionCount << std::endl;		
		#endif

		Detection detections[MAX_DETECTIONS];
		cudaMemcpy(&detections, _devDetections, detectionCount * sizeof(Detection), cudaMemcpyDeviceToHost);
		cudaMemcpy(tmp.data, _devPyramidImage[WB_MAX_OCTAVE_INDEX], _pyramid.canvasImageSize[WB_MAX_OCTAVE_INDEX] * sizeof(float), cudaMemcpyDeviceToHost);

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

		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
			cudaDestroyTextureObject(_texturePyramid[oct]);

		cudaFree(_devOriginalImage);
		cudaFree(_devWorkingImage);

		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
			cudaMalloc((void**)&_devPyramidImage[oct], sizeof(float) * _pyramid.canvasImageSize[oct]);

		cudaFree(_devAlphaBuffer);
		cudaFree(_devDetections);
		cudaFree(_devDetectionCount);
		cudaFree(_devSurvivors);
	}
}
