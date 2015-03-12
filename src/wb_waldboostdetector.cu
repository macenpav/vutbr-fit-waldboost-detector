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
		const uint32 globalId = y * PYRAMID.canvasWidth + x;

		if (x < (PYRAMID.canvasWidth - WB_CLASSIFIER_WIDTH) && y < (PYRAMID.canvasHeight - WB_CLASSIFIER_HEIGHT))
		{
			float response = survivors[globalId].response;

			if (response < FINAL_THRESHOLD)
				return;

			const uint32 x = survivors[globalId].x;
			const uint32 y = survivors[globalId].y;

			__syncthreads();

			bool survived = eval(x, y, &response, startStage, STAGE_COUNT);
			if (survived) {
				uint32 pos = atomicInc(detectionCount, WB_MAX_DETECTIONS);
				detections[pos].x = x;
				detections[pos].y = y;
				detections[pos].width = WB_CLASSIFIER_WIDTH;
				detections[pos].height = WB_CLASSIFIER_HEIGHT;
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
		const uint32 globalId = y * PYRAMID.canvasWidth + x;

		localSurvivors[blockId] = 0;

		if (x < (PYRAMID.canvasWidth - WB_CLASSIFIER_WIDTH) && y < (PYRAMID.canvasHeight - WB_CLASSIFIER_HEIGHT))
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

			survivors[globalId].response = WB_BAD_RESPONSE;

			__syncthreads();

			if (survived) {
				uint32 newThreadId = (blockIdx.y*blockDim.y) * PYRAMID.canvasWidth + (blockIdx.x*blockDim.x) + localSurvivors[blockId];
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
		const uint32 globalId = y * PYRAMID.canvasWidth + x;

		localSurvivors[blockId] = 0;

		if (x < (PYRAMID.canvasWidth - WB_CLASSIFIER_WIDTH) && y < (PYRAMID.canvasHeight - WB_CLASSIFIER_HEIGHT))
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

			survivors[globalId].response = WB_BAD_RESPONSE;

			__syncthreads();

			if (survived) {
				uint32 newThreadId = (blockIdx.y*blockDim.y) * PYRAMID.canvasWidth + (blockIdx.x*blockDim.x) + localSurvivors[blockId];
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
		SurvivorData*		survivors)
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
		values[0] = tex2D(texturePyramidImage, x, y);
		x += stage->width;
		values[1] = tex2D(texturePyramidImage, x, y);
		x += stage->width;
		values[2] = tex2D(texturePyramidImage, x, y);
		y += stage->height;
		values[5] = tex2D(texturePyramidImage, x, y);
		y += stage->height;
		values[8] = tex2D(texturePyramidImage, x, y);
		x -= stage->width;
		values[7] = tex2D(texturePyramidImage, x, y);
		x -= stage->width;
		values[6] = tex2D(texturePyramidImage, x, y);
		y -= stage->height;
		values[3] = tex2D(texturePyramidImage, x, y);
		x += stage->width;
		values[4] = tex2D(texturePyramidImage, x, y);
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

	__global__ void firstPyramidKernel(float* outData, float* finalData)
	{					
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		Octave octave = PYRAMID.octaves[0];
		if (x < octave.width && y < octave.height)
		{						
			float res;
			uint8 i = 1;
			
			// find image index
			for (; i < WB_LEVELS_PER_OCTAVE; ++i)
			{
				if (y < octave.images[i].offsetY)
					break;
			}
			uint8 index = i - 1;

			// black if its outside of the image
			// or downsample from the original texture
			if (x > octave.images[index].width)			
				res = 0.f;					
			else
			{
				float origX = static_cast<float>(x /* - oct.offsetX */) / static_cast<float>(octave.images[index].width) * static_cast<float>(octave.images[0].width);
				float origY = static_cast<float>(y - octave.images[index].offsetY) / static_cast<float>(octave.images[index].height) * static_cast<float>(octave.images[0].height);
				res = tex2D(textureWorkingImage, origX, origY);
			}

			outData[y * octave.width + x] = res;	
			finalData[y * PYRAMID.canvasWidth + x] = res;
		}		
	}

	__global__ void pyramidFromPyramidKernel(float* outData, float* finalData, cudaTextureObject_t inData, uint8 level)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		const Octave octave = PYRAMID.octaves[level];

		if (x < octave.width && y < octave.height)
		{
			// x,y in previous octave
			const float prevX = static_cast<float>(x)* 2.f;
			const float prevY = static_cast<float>(y)* 2.f;

			float res = tex2D<float>(inData, prevX, prevY);

			outData[y * octave.width + x] = res;
			finalData[(y + octave.offsetY) * PYRAMID.canvasWidth + (x + octave.offsetX)] = res;
		}
	}

	__global__ void staticPyramidKernel(float* outData)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		// first octave 
		Octave octave0 = PYRAMID.octaves[0];
		uint32 pitch = PYRAMID.canvasWidth;
		if (x < octave0.width && y < octave0.height)
		{
			// find image index
			uint8 i = 1;
			for (; i < WB_LEVELS_PER_OCTAVE; ++i)
			{
				if (y < octave0.images[i].offsetY)
					break;
			}
			uint8 index = i - 1;

			float res;
			if (x > octave0.images[index].width)
				res = 0.f;
			else
			{
				float origX = static_cast<float>(x /* - oct.offsetX */) / static_cast<float>(octave0.images[index].width) * static_cast<float>(octave0.images[0].width);
				float origY = static_cast<float>(y - octave0.images[index].offsetY) / static_cast<float>(octave0.images[index].height) * static_cast<float>(octave0.images[0].height);
				res = tex2D(textureWorkingImage, origX, origY);
			}
			outData[y * pitch + x] = res;
		}
		else
			return; // we can return, other octaves are always smaller				

		// other octaves are downsampled from a previous octave
		for (uint8 oct = 1; oct < WB_OCTAVES; ++oct)
		{
			__syncthreads(); // sync to be sure previous octave is generated

			Octave octave = PYRAMID.octaves[oct];
			Octave prevOctave = PYRAMID.octaves[oct - 1];

			if (x < octave.width && y < octave.height)
			{
				// x,y in previous octave
				uint32 prevX = x << 1;
				uint32 prevY = y << 1;

				prevX += prevOctave.offsetX;
				prevY += prevOctave.offsetY;
				
				outData[(y + octave.offsetY) * pitch + (x + octave.offsetX)] = tex2D(texturePyramidImage, static_cast<float>(prevX), static_cast<float>(prevY));;
			}
			else
				return; // again we can return, other octaves are smaller
		}
	}

	void WaldboostDetector::_pyramidGenSingleTexture()
	{		
		dim3 grid(_pyramid.canvasWidth / _block.x + 1, _pyramid.canvasHeight / _block.y + 1, 1);		
		staticPyramidKernel <<<grid,_block>>>(_devPyramidData);		
	}

	void WaldboostDetector::_pyramidGenBindlessTexture()
	{
		dim3 grid0(_pyramid.octaves[0].width / _block.x + 1, _pyramid.octaves[0].height / _block.y + 1, 1);
		firstPyramidKernel <<<grid0, _block>>>(_devPyramidImage[0], _devPyramidData);

		cudaResourceDesc resourceDesc0;
		memset(&resourceDesc0, 0, sizeof(resourceDesc0));
		resourceDesc0.resType = cudaResourceTypePitch2D;
		resourceDesc0.res.pitch2D.devPtr = _devPyramidImage[0];
		resourceDesc0.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
		resourceDesc0.res.pitch2D.desc.x = 32; // bits per channel
		resourceDesc0.res.pitch2D.width = _pyramid.octaves[0].width;
		resourceDesc0.res.pitch2D.height = _pyramid.octaves[0].height;
		resourceDesc0.res.pitch2D.pitchInBytes = _pyramid.octaves[0].width * sizeof(float);

		cudaTextureDesc textureDesc0;
		memset(&textureDesc0, 0, sizeof(textureDesc0));
		textureDesc0.readMode = cudaReadModeElementType;
		textureDesc0.filterMode = cudaFilterModeLinear;
		textureDesc0.addressMode[0] = cudaAddressModeClamp;
		textureDesc0.addressMode[1] = cudaAddressModeClamp;
		textureDesc0.normalizedCoords = false;

		cudaCreateTextureObject(&_texturePyramidObjects[0], &resourceDesc0, &textureDesc0, NULL);

		for (uint8 oct = 1; oct < WB_OCTAVES; ++oct)
		{
			dim3 grid(_pyramid.octaves[oct].width / _block.x + 1, _pyramid.octaves[oct].height / _block.y + 1, 1);

			pyramidFromPyramidKernel<<<grid, _block>>>(_devPyramidImage[oct], _devPyramidData, _texturePyramidObjects[oct - 1], oct);

			cudaResourceDesc resourceDesc;
			memset(&resourceDesc, 0, sizeof(resourceDesc));
			resourceDesc.resType = cudaResourceTypePitch2D;
			resourceDesc.res.pitch2D.devPtr = _devPyramidImage[oct];
			resourceDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
			resourceDesc.res.pitch2D.desc.x = 32; // bits per channel
			resourceDesc.res.pitch2D.width = _pyramid.octaves[oct].width;
			resourceDesc.res.pitch2D.height = _pyramid.octaves[oct].height;
			resourceDesc.res.pitch2D.pitchInBytes = _pyramid.octaves[oct].width * sizeof(float);

			cudaTextureDesc textureDesc;
			memset(&textureDesc, 0, sizeof(textureDesc));
			textureDesc.readMode = cudaReadModeElementType;
			textureDesc.filterMode = cudaFilterModeLinear;
			textureDesc.addressMode[0] = cudaAddressModeClamp;
			textureDesc.addressMode[1] = cudaAddressModeClamp;
			textureDesc.normalizedCoords = false;

			cudaCreateTextureObject(&_texturePyramidObjects[oct], &resourceDesc, &textureDesc, NULL);
		}
	}

	void WaldboostDetector::_pyramidKernelWrapper()
	{
		switch (_pyGenMode)
		{
			case PYGEN_SINGLE_TEXTURE:
				_pyramidGenSingleTexture();
				break;

			case PYGEN_BINDLESS_TEXTURE:
				_pyramidGenBindlessTexture();
				break;

			default:
				break;
		}					
	}

	void WaldboostDetector::_precalcHorizontalPyramid()
	{		
		uint32 currentOffsetX = 0;		
		
		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
		{		
			Octave octave;

			octave.width = _info.width >> oct;			
			_pyramid.canvasWidth += octave.width;

			float scale = pow(2.f, oct);
			uint32 currentOffsetY = 0;			
			for (uint8 lvl = 0; lvl < WB_LEVELS_PER_OCTAVE; ++lvl)
			{
				PyramidImage data;

				float scaledWidth = _info.width / scale;
				float scaledHeight = _info.height / scale;

				data.width = static_cast<uint32>(scaledWidth);
				data.height = static_cast<uint32>(scaledHeight);
				data.offsetY = currentOffsetY;
				data.offsetX = octave.offsetX;

				scale *= WB_SCALING_FACTOR;
				currentOffsetY += data.height;				

				octave.images[lvl] = data;
			}

			if (oct == 0) {
				_pyramid.canvasHeight = currentOffsetY;				
			}			

			octave.height = currentOffsetY;
			octave.imageSize = octave.height * octave.width;			
			currentOffsetX += octave.width;

			_pyramid.octaves[oct] = octave;
		}
		
		_pyramid.canvasImageSize = _pyramid.canvasWidth * _pyramid.canvasHeight;
	}

	void WaldboostDetector::_precalc4x8Pyramid()
	{
		_pyramid.canvasWidth = _info.width + (_info.width >> 1);

		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
		{
			Octave octave;

			octave.width = _info.width >> oct;

			float scale = pow(2.f, oct);
			switch (oct)
			{
				case 0:
					octave.offsetX = 0;
					octave.offsetY = 0;
					break;
				case 1:
					octave.offsetX = _pyramid.octaves[0].width;
					octave.offsetY = 0;
					break;
				case 2:
					octave.offsetX = _pyramid.octaves[1].offsetX;
					octave.offsetY = _pyramid.octaves[1].height;
					break;
				case 3:
					octave.offsetX = _pyramid.octaves[2].offsetX + _pyramid.octaves[2].width;
					octave.offsetY = _pyramid.octaves[2].offsetY;
					break;
				default:
					break;
			}

			uint32 currentOffsetY = octave.offsetY;
			uint32 currentHeight = 0;
			for (uint8 lvl = 0; lvl < WB_LEVELS_PER_OCTAVE; ++lvl)
			{
				PyramidImage data;

				float scaledWidth = _info.width / scale;
				float scaledHeight = _info.height / scale;

				data.width = static_cast<uint32>(scaledWidth);
				data.height = static_cast<uint32>(scaledHeight);
				data.offsetY = currentOffsetY;
				data.offsetX = octave.offsetX;

				scale *= WB_SCALING_FACTOR;
				currentOffsetY += data.height;
				currentHeight += data.height;

				octave.images[lvl] = data;
			}

			octave.height = currentHeight;
			octave.imageSize = octave.width * octave.height;

			_pyramid.octaves[oct] = octave;
		}

		_pyramid.canvasHeight = _pyramid.octaves[0].height;
		_pyramid.canvasImageSize = _pyramid.canvasWidth * _pyramid.canvasHeight;		
	}

	void WaldboostDetector::init(cv::Mat* image)
	{
		_info.width = image->cols;
		_info.height = image->rows;
		_info.imageSize = image->cols * image->rows;		
		_info.channels = image->channels();		

		switch (_pyType)
		{
			case PYTYPE_OPTIMIZED:
				_precalc4x8Pyramid();
				break;

			case PYTYPE_HORIZONAL:
				_precalcHorizontalPyramid();
				break;
		}
		

		if (_opt & OPT_VERBOSE)
		{
			std::cout << LIBHEADER << "Finished generating pyramid." << std::endl;
			std::cout << LIBHEADER << "Outputting details ..." << std::endl;
			std::cout << LIBHEADER << "Canvas size: " << _pyramid.canvasImageSize << " (" << _pyramid.canvasWidth << "x" << _pyramid.canvasHeight << ")" << std::endl;
			std::cout << LIBHEADER << "Octaves ..." << std::endl;

			for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
			{
				Octave octave = _pyramid.octaves[oct];
				std::cout << LIBHEADER << "[" << (uint32)oct << "]" << std::endl;
				std::cout << LIBHEADER << "Octave size: " << octave.imageSize << " (" << octave.width << "x" << octave.height << ")" << std::endl;
				std::cout << LIBHEADER << "Octave offsets x: " << octave.offsetX << " y: " << octave.offsetY << std::endl;
				std::cout << LIBHEADER << "Images ..." << std::endl;

				for (uint8 lvl = 0; lvl < WB_LEVELS_PER_OCTAVE; ++lvl)
				{
					PyramidImage im = _pyramid.octaves[oct].images[lvl];
					std::cout << LIBHEADER << "[" << (uint32)oct << "]" << "[" << (uint32)lvl << "]" << std::endl;
					std::cout << LIBHEADER << "Image size: " << im.width << "x" << im.height << std::endl;
					std::cout << LIBHEADER << "Image offsets x: " << im.offsetX << " y: " << im.offsetY << std::endl;
				}
				std::cout << std::endl;
			}
		}

		cudaMemcpyToSymbol(devPyramid, &_pyramid, sizeof(Pyramid));
		cudaMemcpyToSymbol(devInfo, &_info, sizeof(ImageInfo));
		cudaMemcpyToSymbol(stages, hostStages, sizeof(Stage) * STAGE_COUNT);

		cudaMalloc((void**)&_devOriginalImage, sizeof(uint8) * _info.imageSize * _info.channels);
		cudaMalloc((void**)&_devWorkingImage, sizeof(float) * _info.imageSize);
				
		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
			cudaMalloc((void**)&_devPyramidImage[oct], sizeof(float) * _pyramid.octaves[oct].imageSize);		

		cudaMalloc((void**)&_devPyramidData, sizeof(float) * _pyramid.canvasImageSize);

		cudaMalloc((void**)&_devDetections, sizeof(Detection) * WB_MAX_DETECTIONS);
		cudaMalloc((void**)&_devDetectionCount, sizeof(uint32));
		cudaMemset((void**)&_devDetectionCount, 0, sizeof(uint32));
		cudaMalloc((void**)&_devSurvivors, sizeof(SurvivorData) * _pyramid.canvasImageSize);
				
		cudaMalloc(&_devAlphaBuffer, STAGE_COUNT * ALPHA_COUNT * sizeof(float));
		cudaMemcpy(_devAlphaBuffer, alphas, STAGE_COUNT * ALPHA_COUNT * sizeof(float), cudaMemcpyHostToDevice);
		
		cudaChannelFormatDesc alphaDesc = cudaCreateChannelDesc<float>();
		cudaBindTexture(nullptr, &textureAlphas, _devAlphaBuffer, &alphaDesc, STAGE_COUNT * ALPHA_COUNT * sizeof(float));

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(nullptr, &textureWorkingImage, _devWorkingImage, &channelDesc, _info.width, _info.height, sizeof(float) * _info.width);

		cudaChannelFormatDesc finalDesc = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(nullptr, &texturePyramidImage, _devPyramidData, &finalDesc, _pyramid.canvasWidth, _pyramid.canvasHeight, sizeof(float) * _pyramid.canvasWidth);

		#ifdef WB_DEBUG
		_frameCount = 0;
		#endif
	}

	__global__ void clearKernel(float* data, uint32 width, uint32 height)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < width && y < height)
			data[y * width + x] = 0.f;
	}

	__global__ void copyKernel(float* out, cudaTextureObject_t obj, uint32 width, uint32 height)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < width && y < height)
		{
			out[y * width + x] = tex2D<float>(obj, x + 0.5f, y + 0.5f);
		}
	}

	__global__ void copyKernel(float* out, uint32 width, uint32 height)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < width && y < height)
		{
			out[y * width + x] = tex2D(texturePyramidImage, x + 0.5f, y + 0.5f);
		}
	}

	void WaldboostDetector::setImage(cv::Mat* image)
	{
		_initTimers();

		textureWorkingImage.addressMode[0] = cudaAddressModeClamp;
		textureWorkingImage.addressMode[1] = cudaAddressModeClamp;
		textureWorkingImage.filterMode = cudaFilterModeLinear;
		textureWorkingImage.normalized = false;

		texturePyramidImage.addressMode[0] = cudaAddressModeClamp;
		texturePyramidImage.addressMode[1] = cudaAddressModeClamp;
		texturePyramidImage.filterMode = cudaFilterModeLinear;
		texturePyramidImage.normalized = false;

		_myImage = image;
		cudaMemcpy(_devOriginalImage, image->data, _info.imageSize * _info.channels * sizeof(uint8), cudaMemcpyHostToDevice);

		dim3 grid(_info.width / _block.x + 1, _info.height / _block.y + 1, 1);

		cudaEvent_t start_preprocess, stop_preprocess;
		if (_opt & OPT_TIMER)
		{
			cudaEventCreate(&start_preprocess);
			cudaEventCreate(&stop_preprocess);
			cudaEventRecord(start_preprocess);
		}

		preprocessKernel <<<grid, _block>>>(_devWorkingImage, _devOriginalImage);

		if (_opt & OPT_TIMER)
		{
			cudaEventRecord(stop_preprocess);
			cudaEventSynchronize(stop_preprocess);
			cudaEventElapsedTime(&_timers[TIMER_PREPROCESS], start_preprocess, stop_preprocess);
		}

		if (_opt & OPT_VISUAL_DEBUG)
		{
			cv::Mat tmp(cv::Size(_info.width, _info.height), CV_32FC1);
			cudaMemcpy(tmp.data, _devWorkingImage, _info.imageSize * sizeof(float), cudaMemcpyDeviceToHost);
			cv::imshow("Preprocessed image (B&W image should be displayed)", tmp);
			cv::waitKey(WB_WAIT_DELAY);
		}

		cudaEvent_t start_pyramid, stop_pyramid;		
		if (_opt & OPT_TIMER)
		{
			cudaEventCreate(&start_pyramid);
			cudaEventCreate(&stop_pyramid);
			cudaEventRecord(start_pyramid);
		}

		_pyramidKernelWrapper();
		
		if (_opt & OPT_TIMER)
		{
			cudaEventRecord(stop_pyramid);
			cudaEventSynchronize(stop_pyramid);
			cudaEventElapsedTime(&_timers[TIMER_PYRAMID], start_pyramid, stop_pyramid);
		}
		
		if (_opt & OPT_VISUAL_DEBUG)
		{	
			// copy image from GPU to CPU
			float* pyramidImage;			
			dim3 grid(_pyramid.canvasWidth / _block.x + 1, _pyramid.canvasHeight / _block.y + 1, 1);
			cudaMalloc((void**)&pyramidImage, _pyramid.canvasImageSize * sizeof(float));
			
			// copies from statically defined texture
			copyKernel <<<grid, _block>>>(pyramidImage, _pyramid.canvasWidth, _pyramid.canvasHeight);			

			// display using OpenCV
			cv::Mat tmp(cv::Size(_pyramid.canvasWidth, _pyramid.canvasHeight), CV_32FC1);
			cudaMemcpy(tmp.data, pyramidImage, _pyramid.canvasImageSize * sizeof(float), cudaMemcpyDeviceToHost);			
			cv::imshow("Pyramid texture (B&W pyramid images should be displayed)", tmp);
			cv::waitKey(WB_WAIT_DELAY);			

			cudaFree(pyramidImage);
		}
	}

	void WaldboostDetector::_initTimers()
	{
		for (uint8 i = 0; i < MAX_TIMERS; ++i)
			_timers[i] = 0.f;
	}

	void WaldboostDetector::_processDetections()
	{
		Detection detections[WB_MAX_DETECTIONS];
		uint32 detectionCount = 0;
		cudaMemcpy(&detectionCount, _devDetectionCount, sizeof(uint32), cudaMemcpyDeviceToHost);
		cudaMemcpy(&detections, _devDetections, detectionCount * sizeof(Detection), cudaMemcpyDeviceToHost);

		if (_opt & OPT_VERBOSE)
		{			
			std::cout << LIBHEADER << "Detections: " << detectionCount << std::endl;
		}

		for (uint32 i = 0; i < detectionCount; ++i)
		{
			Detection d = detections[i];

			uint8 oct, lvl;
			for (oct = 0; oct < WB_OCTAVES; ++oct)
			{
				Octave octave = _pyramid.octaves[oct];
				if (d.x >= octave.offsetX && d.x < octave.offsetX + octave.width && 
					d.y >= octave.offsetY && d.y < octave.offsetY + octave.height)
				{
					for (lvl = 0; lvl < WB_LEVELS_PER_OCTAVE; ++lvl)
					{
						PyramidImage pi = octave.images[lvl];
						if (d.x >= pi.offsetX && d.x < pi.offsetX + pi.width &&
							d.y >= pi.offsetY && d.y < pi.offsetY + pi.height)
							break;
					}
					break;
				}
			}

			float scale = pow(2.f, static_cast<float>(oct)+1.f / static_cast<float>(WB_LEVELS_PER_OCTAVE)* static_cast<float>(lvl));

			d.width = static_cast<uint32>(static_cast<float>(d.width) * scale);
			d.height = static_cast<uint32>(static_cast<float>(d.height) * scale);

			d.x -= (_pyramid.octaves[oct].images[lvl].offsetX);
			d.x = static_cast<uint32>(static_cast<float>(d.x) * scale);

			d.y -= (_pyramid.octaves[oct].images[lvl].offsetY);
			d.y = static_cast<uint32>(static_cast<float>(d.y) * scale);

			if (_opt & (OPT_VISUAL_DEBUG|OPT_VISUAL_OUTPUT))
				cv::rectangle(*_myImage, cvPoint(d.x, d.y), cvPoint(d.x + d.width, d.y + d.height), CV_RGB(0, 255, 0));
		}	
	}

	void WaldboostDetector::run()
	{		
		if (_opt & OPT_VERBOSE)		
			std::cout << LIBHEADER << "Processing detections ..." << std::endl;		
		
		dim3 grid(_pyramid.canvasWidth / _block.x + 1, _pyramid.canvasHeight / _block.y + 1, 1);
		cudaMemset(_devDetectionCount, 0, sizeof(uint32));
		
		cudaEvent_t start_detection, stop_detection;
		if (_opt & OPT_TIMER)
		{			
			cudaEventCreate(&start_detection);
			cudaEventCreate(&stop_detection);
			cudaEventRecord(start_detection);
		}

		detectionKernel <<<grid, _block>>>(_devDetections, _devDetectionCount, _devSurvivors);
		
		if (_opt & OPT_TIMER)
		{
			cudaEventRecord(stop_detection);
			cudaEventSynchronize(stop_detection);
			cudaEventElapsedTime(&_timers[TIMER_DETECTION], start_detection, stop_detection);
		}						

		_processDetections();	

		if (_opt & OPT_TIMER)
		{			
			if (_opt & OPT_VERBOSE)
			{ 
				std::cout << "Preprocessing: " << _timers[TIMER_PREPROCESS] << std::endl;
				std::cout << "Pyramid gen.: " << _timers[TIMER_PYRAMID] << std::endl;
				std::cout << "Detection: " << _timers[TIMER_DETECTION] << std::endl;
			}

			if (_opt & (OPT_VISUAL_OUTPUT | OPT_VISUAL_DEBUG))
			{
				std::string t1 = std::string("Preprocessing: ") + std::to_string(_timers[TIMER_PREPROCESS]) + std::string(" ms");
				std::string t2 = std::string("Pyramid gen.: ") + std::to_string(_timers[TIMER_PYRAMID]) + std::string(" ms");
				std::string t3 = std::string("Detection: ") + std::to_string(_timers[TIMER_DETECTION]) + std::string(" ms");
				cv::putText(*_myImage, t1, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.35, CV_RGB(0, 255, 0));
				cv::putText(*_myImage, t2, cv::Point(10, 35), cv::FONT_HERSHEY_SIMPLEX, 0.35, CV_RGB(0, 255, 0));
				cv::putText(*_myImage, t3, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.35, CV_RGB(0, 255, 0));
			}
		}
	}

	void WaldboostDetector::free()
	{
		cudaUnbindTexture(textureWorkingImage);
		cudaUnbindTexture(textureAlphas);

		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
		{ 
			cudaFree(_devPyramidImage);
			cudaDestroyTextureObject(_texturePyramidObjects[oct]);
		}
			
		cudaFree(_devOriginalImage);
		cudaFree(_devWorkingImage);	
		cudaFree(_devPyramidImage);
		cudaFree(_devAlphaBuffer);
		cudaFree(_devDetections);
		cudaFree(_devDetectionCount);
		cudaFree(_devSurvivors);
	}
}
