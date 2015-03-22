#include "wbd_waldboostdetector.cuh"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include "cuda_runtime.h"

#include "wbd_general.h"
#include "wbd_structures.h"
#include "wbd_detector.h"
#include "wbd_alphas.h"
#include "wbd_simple.h"
#include "wbd_func.h"

namespace wbd 
{
		
	__device__ 
		void detectDetections_prefixsum(
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		uint32 const& threadId,
		uint32 const&	globalOffset,
		SurvivorData*	survivors,
		Detection*		detections,
		uint32*			detectionCount,
		uint16			startStage)
	{				
		float response = survivors[globalOffset + threadId].response;
		const uint32 x = survivors[globalOffset + threadId].x;
		const uint32 y = survivors[globalOffset + threadId].y;

		bool survived = eval(texture, alphas, x, y, &response, startStage, WB_STAGE_COUNT);
		if (survived) {
			uint32 pos = atomicInc(detectionCount, WB_MAX_DETECTIONS);
			detections[pos].x = x;
			detections[pos].y = y;
			detections[pos].width = WB_CLASSIFIER_WIDTH;
			detections[pos].height = WB_CLASSIFIER_HEIGHT;
			detections[pos].response = response;
		}		
	}

	__device__
		void detectDetections_atomicShared(
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		uint32 const&	threadId,
		SurvivorData*	localSurvivors,		
		Detection*		detections,
		uint32*			detectionCount,
		uint16			startStage)
	{				
		float response = localSurvivors[threadId].response;
		const uint32 x = localSurvivors[threadId].x;
		const uint32 y = localSurvivors[threadId].y;		

		bool survived = eval(texture, alphas, x, y, &response, startStage, WB_STAGE_COUNT);
		if (survived) 
		{
			uint32 pos = atomicInc(detectionCount, WB_MAX_DETECTIONS);
			detections[pos].x = x;
			detections[pos].y = y;
			detections[pos].width = WB_CLASSIFIER_WIDTH;
			detections[pos].height = WB_CLASSIFIER_HEIGHT;
			detections[pos].response = response;
		}		
	}

	__device__
		void detectDetections_global(
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		uint32 const&	threadId,
		uint32 const&	globalOffset,
		SurvivorData*	globalSurvivors,		
		Detection*		detections,
		uint32*			detectionCount,
		uint16			startStage)
	{		
		const uint32 id = globalOffset + threadId;

		float response = globalSurvivors[id].response;
		const uint32 x = globalSurvivors[id].x;
		const uint32 y = globalSurvivors[id].y;		

		bool survived = eval(texture, alphas, x, y, &response, startStage, WB_STAGE_COUNT);
		if (survived)
		{
			uint32 pos = atomicInc(detectionCount, WB_MAX_DETECTIONS);
			detections[pos].x = x;
			detections[pos].y = y;
			detections[pos].width = WB_CLASSIFIER_WIDTH;
			detections[pos].height = WB_CLASSIFIER_HEIGHT;
			detections[pos].response = response;
		}
	}

	__device__ 
	void detectSurvivorsInit_prefixsum(
	cudaTextureObject_t texture,
	cudaTextureObject_t alphas,
		uint32 const&	x,
		uint32 const&	y,		
		uint32 const&	threadId,
		uint32 const&	globalOffset,
		uint32 const&	blockSize,		
		SurvivorData*	survivors,
		uint32&			survivorCount,
		uint32*			survivorScanArray,
		uint16			endStage)
	{				
			
		float response = 0.0f;
		bool survived = eval(texture, alphas, x, y, &response, 0, endStage);

		survivorScanArray[threadId] = survived ? 1 : 0;			

		__syncthreads();

		// up-sweep
		uint32 offset = 1;
		for (uint32 d = blockSize >> 1; d > 0; d >>= 1, offset <<= 1) 
		{
			__syncthreads();

			if (threadId < d)
			{
				const uint32 ai = offset * (2 * threadId + 1) - 1;
				const uint32 bi = offset * (2 * threadId + 2) - 1;
				survivorScanArray[bi] += survivorScanArray[ai];
			}
		}

		// down-sweep
		if (threadId == 0) {
			survivorScanArray[blockSize - 1] = 0;
		}

		for (uint32 d = 1; d < blockSize; d <<= 1) 
		{
			offset >>= 1;

			__syncthreads();

			if (threadId < d)
			{
				const uint32 ai = offset * (2 * threadId + 1) - 1;
				const uint32 bi = offset * (2 * threadId + 2) - 1;

				const uint32 t = survivorScanArray[ai];
				survivorScanArray[ai] = survivorScanArray[bi];
				survivorScanArray[bi] += t;
			}
		}
			
		__syncthreads();		

		if (threadId == 0)
			survivorCount = survivorScanArray[blockSize - 1];
	
		if (survived)
		{			
			uint32 newThreadId = survivorScanArray[threadId];

			// save position and current response
			survivors[newThreadId].x = x;
			survivors[newThreadId].y = y;
			survivors[newThreadId].response = response;
		}				
	}

	__device__
		void detectSurvivorsInit_atomicShared(
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		uint32 const&	x,
		uint32 const&	y,
		uint32 const&	threadId,
		SurvivorData*	localSurvivors,
		uint32*			localSurvivorCount,
		uint16			endStage)
	{				
		float response = 0.0f;
		bool survived = eval(texture, alphas, x, y, &response, 0, endStage);						
		if (survived)
		{				
			uint32 newThreadId = atomicInc(localSurvivorCount, blockDim.x * blockDim.y);
			// save position and current response
			localSurvivors[newThreadId].x = x;
			localSurvivors[newThreadId].y = y;
			localSurvivors[newThreadId].response = response;
		}	
	}

	__device__
		void detectSurvivorsInit_global(	
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		uint32 const&	x,
		uint32 const&	y,
		uint32 const&	threadId,
		uint32 const&	globalOffset,
		SurvivorData*	globalSurvivors,
		uint32*			survivorCount,
		uint16			endStage)
	{						
		float response = 0.0f;
		bool survived = eval(texture, alphas, x, y, &response, 0, endStage);

		if (survived)
		{			
			uint32 threadOffset = atomicInc(survivorCount, blockDim.x * blockDim.y); // there can be max. block size survivors
			uint32 newThreadId = globalOffset + threadOffset;
			// save position and current response
			globalSurvivors[newThreadId].x = x;
			globalSurvivors[newThreadId].y = y;
			globalSurvivors[newThreadId].response = response;
		}	
	}

	__device__ void detectSurvivors_prefixsum(
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		uint32 const&		threadId,
		uint32 const&		globalOffset,
		uint32 const&		blockSize,
		SurvivorData*		survivors,
		uint32&				survivorCount,
		uint32*				survivorScanArray,		
		uint16				startStage,
		uint16				endStage)
	{					
		float response = survivors[globalOffset + threadId].response;			
		const uint32 x = survivors[globalOffset + threadId].x;
		const uint32 y = survivors[globalOffset + threadId].y;

		bool survived = eval(texture, alphas, x, y, &response, startStage, endStage);		
		survivorScanArray[threadId] = survived ? 1 : 0;

		// up-sweep
		int offset = 1;
		for (uint32 d = blockSize >> 1; d > 0; d >>= 1, offset <<= 1) {
			__syncthreads();

			if (threadId < d) {
				uint32 ai = offset * (2 * threadId + 1) - 1;
				uint32 bi = offset * (2 * threadId + 2) - 1;
				survivorScanArray[bi] += survivorScanArray[ai];
			}
		}

		// down-sweep
		if (threadId == 0) {
			survivorScanArray[blockSize - 1] = 0;
		}

		for (uint32 d = 1; d < blockSize; d <<= 1) {
			offset >>= 1;

			__syncthreads();

			if (threadId < d) {
				uint32 ai = offset * (2 * threadId + 1) - 1;
				uint32 bi = offset * (2 * threadId + 2) - 1;

				uint32 t = survivorScanArray[ai];
				survivorScanArray[ai] = survivorScanArray[bi];
				survivorScanArray[bi] += t;
			}
		}		

		__syncthreads();

		if (threadId == 0)
			survivorCount = survivorScanArray[blockSize - 1];

		if (survived) {
			uint32 newThreadId = globalOffset + survivorScanArray[threadId];
			// save position and current response
			survivors[newThreadId].x = x;
			survivors[newThreadId].y = y;
			survivors[newThreadId].response = response;
		}	
	}

	__device__ void detectSurvivors_atomicShared(
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		uint32 const&	threadId,
		SurvivorData*	localSurvivors,
		uint32*			localSurvivorCount,
		uint16			startStage,
		uint16			endStage)
	{										
		float response = localSurvivors[threadId].response;
		const uint32 x = localSurvivors[threadId].x;
		const uint32 y = localSurvivors[threadId].y;

		bool survived = eval(texture, alphas, x, y, &response, startStage, endStage);
		if (survived)
		{
			uint32 newThreadId = atomicInc(localSurvivorCount, blockDim.x * blockDim.y);
			localSurvivors[newThreadId].x = x;
			localSurvivors[newThreadId].y = y;
			localSurvivors[newThreadId].response = response;
		}	
	}

	__device__ void detectSurvivors_global(
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		uint32 const&	threadId,
		uint32 const&	globalOffset,
		SurvivorData*	globalSurvivors,
		uint32*			survivorCount,
		uint16			startStage,
		uint16			endStage)
	{				
		const uint32 id = globalOffset + threadId;

		float response = globalSurvivors[id].response;
		const uint32 x = globalSurvivors[id].x;
		const uint32 y = globalSurvivors[id].y;
		
		bool survived = eval(texture, alphas, x, y, &response, startStage, endStage);
		if (survived)
		{
			uint32 threadOffset = atomicInc(survivorCount, blockDim.x * blockDim.y); // there can be max. block size survivors
			uint32 newThreadId = globalOffset + threadOffset;
			globalSurvivors[newThreadId].x = x;
			globalSurvivors[newThreadId].y = y;
			globalSurvivors[newThreadId].response = response;
		}
	}

	__global__ void detectionKernel_atomicShared(
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		Detection*			detections,
		uint32*				detectionCount)
	{		
		extern __shared__ SurvivorData survivors[];
		__shared__ uint32 survivorCount;
				
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < PYRAMID.canvasWidth - WB_CLASSIFIER_WIDTH && y < PYRAMID.canvasHeight - WB_CLASSIFIER_HEIGHT)
		{	
			const uint32 threadId = threadIdx.y * blockDim.x + threadIdx.x;

			if (threadId == 0)
				survivorCount = 0;
			__syncthreads();

			detectSurvivorsInit_atomicShared(texture, alphas, x, y, threadId, survivors, &survivorCount, 1);

			__syncthreads();
			if (threadId >= survivorCount)
				return;
			__syncthreads();
			if (threadId == 0)
				survivorCount = 0;
			__syncthreads();

			detectSurvivors_atomicShared(texture, alphas, threadId, survivors, &survivorCount, 1, 8);

			__syncthreads();
			if (threadId >= survivorCount)
				return;
			__syncthreads();
			if (threadId == 0)
				survivorCount = 0;
			__syncthreads();

			detectSurvivors_atomicShared(texture, alphas, threadId, survivors, &survivorCount, 8, 64);

			__syncthreads();
			if (threadId >= survivorCount)
				return;
			__syncthreads();
			if (threadId == 0)
				survivorCount = 0;
			__syncthreads();

			detectSurvivors_atomicShared(texture, alphas, threadId, survivors, &survivorCount, 64, 256);

			__syncthreads();
			if (threadId >= survivorCount)
				return;
			__syncthreads();
			if (threadId == 0)
				survivorCount = 0;
			__syncthreads();

			detectSurvivors_atomicShared(texture, alphas, threadId, survivors, &survivorCount, 256, 512);

			__syncthreads();
			if (threadId >= survivorCount)
				return;		
			__syncthreads();

			detectDetections_atomicShared(texture, alphas, threadId, survivors, detections, detectionCount, 512);
		}
	}

	__global__ void detectionKernel_global(
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		SurvivorData*		survivors,		
		Detection*			detections,
		uint32*				detectionCount)
	{		
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < PYRAMID.canvasWidth - WB_CLASSIFIER_WIDTH && y < PYRAMID.canvasHeight - WB_CLASSIFIER_HEIGHT)
		{
			__shared__ uint32 blockSurvivors;

			const uint32 blockSize = blockDim.x * blockDim.y;
			const uint32 blockPitch = gridDim.x * blockSize;

			// every block has a reserved space in global mem.
			const uint32 blockOffset = blockIdx.y * blockPitch + blockIdx.x * blockSize;
			// thread id inside a block
			const uint32 threadId = threadIdx.y * blockDim.x + threadIdx.x;
			
			if (threadId == 0)
				blockSurvivors = 0;

			__syncthreads();

			detectSurvivorsInit_global(texture, alphas, x, y, threadId, blockOffset, survivors, &blockSurvivors, 1);

			// finish all the detections within a block
			__syncthreads();
			if (threadId >= blockSurvivors)
				return;
			// dump all threads, which didn't survive
			__syncthreads();			

			if (threadId == 0)
				blockSurvivors = 0;
			// reset the counter
			__syncthreads();

			detectSurvivors_global(texture, alphas, threadId, blockOffset, survivors, &blockSurvivors, 1, 8);

			// finish all the detections within a block
			__syncthreads();
			if (threadId >= blockSurvivors)
				return;
			// dump all threads, which didn't survive
			__syncthreads();
			if (threadId == 0)
				blockSurvivors = 0;
			// reset the counter
			__syncthreads();

			detectSurvivors_global(texture, alphas, threadId, blockOffset, survivors, &blockSurvivors, 8, 64);

			// finish all the detections within a block
			__syncthreads();
			if (threadId >= blockSurvivors)
				return;
			// dump all threads, which didn't survive
			__syncthreads();
			if (threadId == 0)
				blockSurvivors = 0;
			// reset the counter
			__syncthreads();

			detectSurvivors_global(texture, alphas, threadId, blockOffset, survivors, &blockSurvivors, 64, 256);

			// finish all the detections within a block
			__syncthreads();
			if (threadId >= blockSurvivors)
				return;
			// dump all threads, which didn't survive
			__syncthreads();
			if (threadId == 0)
				blockSurvivors = 0;
			// reset the counter
			__syncthreads();

			detectSurvivors_global(texture, alphas, threadId, blockOffset, survivors, &blockSurvivors, 256, 512);

			// finish all the detections within a block
			__syncthreads();
			if (threadId >= blockSurvivors)
				return;

			detectDetections_global(texture, alphas, threadId, blockOffset, survivors, detections, detectionCount, 512);
		}
	}

	__global__ void detectionKernel_prefixsum(	
		cudaTextureObject_t texture,
		cudaTextureObject_t alphas,
		Detection*			detections,
		uint32*				detectionCount)
	{										
		extern __shared__ SurvivorData survivors[];
		uint32* survivorScanArray = (uint32*)&survivors[blockDim.x * blockDim.y];

		__shared__ uint32 survivorCount;

		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;	
	
		const uint32 blockSize = blockDim.x * blockDim.y;			
		const uint32 blockPitch = gridDim.x * blockSize;			
		const uint32 blockOffset = blockIdx.y * blockPitch + blockIdx.x * blockSize;
		const uint32 threadId = threadIdx.y * blockDim.x + threadIdx.x;

		if (threadId == 0)
			survivorCount = 0;

		__syncthreads();

		detectSurvivorsInit_prefixsum(texture, alphas, x, y, threadId, blockOffset, blockSize, survivors, survivorCount, survivorScanArray, 1);
			
			__syncthreads();			
			if (threadId >= survivorCount)
				return;
			__syncthreads();			
			if (threadId == 0)
				survivorCount = 0;
			__syncthreads();

			detectSurvivors_atomicShared(texture, alphas, threadId, survivors, &survivorCount, 1, 8);
			
			__syncthreads();
			if (threadId >= survivorCount)
				return;
			__syncthreads();
			if (threadId == 0)
				survivorCount = 0;
			__syncthreads();
			detectSurvivors_atomicShared(texture, alphas, threadId, survivors, &survivorCount, 8, 64);
			
			
			__syncthreads();
			if (threadId >= survivorCount)
				return;
			__syncthreads();
			if (threadId == 0)
				survivorCount = 0;
			__syncthreads();
			
			detectSurvivors_atomicShared(texture, alphas, threadId, survivors, &survivorCount, 64, 256);
			
			__syncthreads();
			if (threadId >= survivorCount)
				return;
			__syncthreads();
			if (threadId == 0)
				survivorCount = 0;
			__syncthreads();

			detectSurvivors_atomicShared(texture, alphas, threadId, survivors, &survivorCount, 256, 512);
			
			__syncthreads();
			if (threadId >= survivorCount)
				return;

			detectDetections_atomicShared(texture, alphas, threadId, survivors, detections, detectionCount, 512);
		
	}

	__device__ void sumRegions(cudaTextureObject_t texture, float* values, float x, float y, Stage* stage)
	{
		values[0] = tex2D<float>(texture, x, y);
		x += stage->width;
		values[1] = tex2D<float>(texture, x, y);
		x += stage->width;
		values[2] = tex2D<float>(texture, x, y);
		y += stage->height;
		values[5] = tex2D<float>(texture, x, y);
		y += stage->height;
		values[8] = tex2D<float>(texture, x, y);
		x -= stage->width;
		values[7] = tex2D<float>(texture, x, y);
		x -= stage->width;
		values[6] = tex2D<float>(texture, x, y);
		y -= stage->height;
		values[3] = tex2D<float>(texture, x, y);
		x += stage->width;
		values[4] = tex2D<float>(texture, x, y);
	}

	__device__ float evalLBP(cudaTextureObject_t texture, cudaTextureObject_t alphas, uint32 x, uint32 y, Stage* stage)
	{
		const uint8 LBPOrder[8] = { 0, 1, 2, 5, 8, 7, 6, 3 };

		float values[9];

		sumRegions(texture, values, static_cast<float>(x) + (static_cast<float>(stage->width) * 0.5f), y + (static_cast<float>(stage->height) * 0.5f), stage);

		uint8 code = 0;
		for (uint8 i = 0; i < 8; ++i)
			code |= (values[LBPOrder[i]] > values[4]) << i;

		return tex1Dfetch<float>(alphas, stage->alphaOffset + code);
	}

	__device__ bool eval(cudaTextureObject_t texture, cudaTextureObject_t alphas, uint32 x, uint32 y, float* response, uint16 startStage, uint16 endStage)
	{
		for (uint16 i = startStage; i < endStage; ++i) {
			Stage stage = stages[i];
			*response += evalLBP(texture, alphas, x + stage.x, y + stage.y, &stage);
			if (*response < stage.thetaB) {
				return false;
			}
		}

		// final waldboost threshold
		return *response > WB_FINAL_THRESHOLD;
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

	__global__ void createFirstPyramid(cudaTextureObject_t initialImageTexture, float* subPyramidImageData, float* finalImageData)
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
				res = tex2D<float>(initialImageTexture, origX, origY);
			}

			subPyramidImageData[y * octave.width + x] = res;
			finalImageData[y * PYRAMID.canvasWidth + x] = res;
		}		
	}

	__global__ void pyramidFromPyramidKernel(float* pyramidLevelImage, float* pyramidFinalImage, cudaTextureObject_t inData, uint8 level)
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

			pyramidLevelImage[y * octave.width + x] = res;
			pyramidFinalImage[(y + octave.offsetY) * PYRAMID.canvasWidth + (x + octave.offsetX)] = res;
		}
	}

	__global__ void createPyramidSingleTexture(float* outImage, cudaTextureObject_t pyramidImageTexture, cudaTextureObject_t inPreprocessedImageTexture)
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
				res = tex2D<float>(inPreprocessedImageTexture, origX, origY);
			}
			outImage[y * pitch + x] = res;
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
				
				outImage[(y + octave.offsetY) * pitch + (x + octave.offsetX)] = tex2D<float>(pyramidImageTexture, static_cast<float>(prevX), static_cast<float>(prevY));;
			}
			else
				return; // again we can return, other octaves are smaller
		}
	}

	void WaldboostDetector::_pyramidGenSingleTexture()
	{		
		dim3 grid(_pyramid.canvasWidth / _block.x + 1, _pyramid.canvasHeight / _block.y + 1, 1);		
		createPyramidSingleTexture <<<grid, _block>>>(_devPyramidData, _finalPyramidTexture, _preprocessedImageTexture);
	}
	
	

	void WaldboostDetector::_pyramidGenBindlessTexture()
	{
		dim3 grid0(_pyramid.octaves[0].width / _block.x + 1, _pyramid.octaves[0].height / _block.y + 1, 1);
		createFirstPyramid << <grid0, _block >> >(_preprocessedImageTexture, _devPyramidImage[0], _devPyramidData);

		bindFloatImageTo2DTexture(&_texturePyramidObjects[0], _devPyramidImage[0], _pyramid.octaves[0].width, _pyramid.octaves[0].height);		

		for (uint8 oct = 1; oct < WB_OCTAVES; ++oct)
		{
			dim3 grid(_pyramid.octaves[oct].width / _block.x + 1, _pyramid.octaves[oct].height / _block.y + 1, 1);

			pyramidFromPyramidKernel<<<grid, _block>>>(_devPyramidImage[oct], _devPyramidData, _texturePyramidObjects[oct - 1], oct);

			if (oct != WB_OCTAVES - 1)			
				bindFloatImageTo2DTexture(&_texturePyramidObjects[oct], _devPyramidImage[oct], _pyramid.octaves[oct].width, _pyramid.octaves[oct].height);
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

		_frame = 0;

		ClockPoint init_time_start, init_time_end;
		if (_opt & OPT_TIMER)
		{ 
			_initTimers();
			init_time_start = Clock::now();
		}
	
		switch (_pyType)
		{
			case PYTYPE_OPTIMIZED:
				_precalc4x8Pyramid();
				break;

			case PYTYPE_HORIZONAL:
				_precalcHorizontalPyramid();
				break;
		}

		if (_opt & OPT_TIMER)
		{
			init_time_end = Clock::now();
			FPDuration duration = init_time_end - init_time_start;
			_timers[TIMER_INIT] += static_cast<float>(std::chrono::duration_cast<Nanoseconds>(duration).count()) / 1000000.f;
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

		if (_opt & OPT_TIMER)
		{
			init_time_start = Clock::now();
		}

		cudaMemcpyToSymbol(devPyramid, &_pyramid, sizeof(Pyramid));
		cudaMemcpyToSymbol(devInfo, &_info, sizeof(ImageInfo));
		cudaMemcpyToSymbol(stages, hostStages, sizeof(Stage) * WB_STAGE_COUNT);

		cudaMalloc((void**)&_devOriginalImage, sizeof(uint8) * _info.imageSize * _info.channels);
		cudaMalloc((void**)&_devPreprocessedImage, sizeof(float) * _info.imageSize);
				
		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
			cudaMalloc((void**)&_devPyramidImage[oct], sizeof(float) * _pyramid.octaves[oct].imageSize);		

		cudaMalloc((void**)&_devPyramidData, sizeof(float) * _pyramid.canvasImageSize);

		cudaMalloc((void**)&_devDetections, sizeof(Detection) * WB_MAX_DETECTIONS);
		cudaMalloc((void**)&_devDetectionCount, sizeof(uint32));
		cudaMemset((void**)&_devDetectionCount, 0, sizeof(uint32));
		cudaMalloc((void**)&_devSurvivors, sizeof(SurvivorData) * (_pyramid.canvasWidth / _block.x + 1) * (_pyramid.canvasHeight / _block.y + 1) * (_block.x * _block.y));
		cudaMalloc((void**)&_devSurvivorCount, sizeof(uint32));
		cudaMemset((void**)&_devSurvivorCount, 0, sizeof(uint32));
				
		cudaMalloc(&_devAlphaBuffer, WB_STAGE_COUNT * WB_ALPHA_COUNT * sizeof(float));
		cudaMemcpy(_devAlphaBuffer, alphas, WB_STAGE_COUNT * WB_ALPHA_COUNT * sizeof(float), cudaMemcpyHostToDevice);
		
		bindLinearFloatDataToTexture(&_alphasTexture, _devAlphaBuffer, WB_STAGE_COUNT * WB_ALPHA_COUNT);

		bindFloatImageTo2DTexture(&_finalPyramidTexture, _devPyramidData, _pyramid.canvasWidth, _pyramid.canvasHeight);

		cudaDeviceSynchronize();

		if (_opt & OPT_TIMER)
		{
			init_time_end = Clock::now();
			FPDuration duration = init_time_end - init_time_start;
			_timers[TIMER_INIT] += static_cast<float>(std::chrono::duration_cast<Nanoseconds>(duration).count()) / 1000000.f;
		}

		if (_opt & OPT_OUTPUT_CSV)
		{
			std::ofstream file;
			file.open(_outputFilename, std::ios::out);
			file << "init;" << _timers[TIMER_INIT] << std::endl << std::endl;
			file << "frame;preprocessing;pyramid gen.;detection" << std::endl;
			file.close();
		}
	}

	__global__ void clearKernel(float* data, uint32 width, uint32 height)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < width && y < height)
			data[y * width + x] = 0.f;
	}

	__global__ void copyImageFromTextureObject(float* out, cudaTextureObject_t obj, uint32 width, uint32 height)
	{
		const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
		const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < width && y < height)
		{
			out[y * width + x] = tex2D<float>(obj, x + 0.5f, y + 0.5f);
		}
	}

	void WaldboostDetector::setImage(cv::Mat* image)
	{
		_initTimers();		

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

		preprocessKernel <<<grid, _block>>>(_devPreprocessedImage, _devOriginalImage);

		if (_opt & OPT_TIMER)
		{
			cudaEventRecord(stop_preprocess);
			cudaEventSynchronize(stop_preprocess);
			cudaEventElapsedTime(&_timers[TIMER_PREPROCESS], start_preprocess, stop_preprocess);
		}

		if (_opt & OPT_VISUAL_DEBUG)
		{
			cv::Mat tmp(cv::Size(_info.width, _info.height), CV_32FC1);
			cudaMemcpy(tmp.data, _devPreprocessedImage, _info.imageSize * sizeof(float), cudaMemcpyDeviceToHost);
			cv::imshow("Preprocessed image (B&W image should be displayed)", tmp);
			cv::waitKey(WB_WAIT_DELAY);
		}

		bindFloatImageTo2DTexture(&_preprocessedImageTexture, _devPreprocessedImage, _info.width, _info.height);

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
			copyImageFromTextureObject<<<grid, _block>>>(pyramidImage, _finalPyramidTexture, _pyramid.canvasWidth, _pyramid.canvasHeight);

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

		_processDetections(detections, detectionCount);
	}

	void WaldboostDetector::_processDetections(Detection* detections, uint32 const& detectionCount)
	{		
		if (_opt & OPT_VERBOSE)	
			std::cout << LIBHEADER << "Detection count: " << detectionCount << std::endl;		

		if (_opt & (OPT_VISUAL_DEBUG|OPT_VISUAL_OUTPUT))
		{ 
			std::string t = std::string("Detection count: ") + std::to_string(detectionCount);
			cv::putText(*_myImage, t, cv::Point(10, 65), cv::FONT_HERSHEY_SIMPLEX, 0.35, CV_RGB(0, 255, 0));
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
		switch (_detectionMode)
		{
			case DET_ATOMIC_GLOBAL:
				if (_opt & OPT_TIMER)
				{
					cudaEventCreate(&start_detection);
					cudaEventCreate(&stop_detection);
					cudaEventRecord(start_detection);
				}
				detectionKernel_global<<<grid, _block>>>(_finalPyramidTexture, _alphasTexture, _devSurvivors, _devDetections, _devDetectionCount);
				if (_opt & OPT_TIMER)
				{
					cudaEventRecord(stop_detection);
					cudaEventSynchronize(stop_detection);
					cudaEventElapsedTime(&_timers[TIMER_DETECTION], start_detection, stop_detection);
				}
				_processDetections();
				break;

			case DET_ATOMIC_SHARED:
			{
				if (_opt & OPT_TIMER)
				{
					cudaEventCreate(&start_detection);
					cudaEventCreate(&stop_detection);
					cudaEventRecord(start_detection);
				}
				uint32 sharedMemsize = _block.x * _block.y * sizeof(SurvivorData);
				detectionKernel_atomicShared << <grid, _block, sharedMemsize >> >(_finalPyramidTexture, _alphasTexture, _devDetections, _devDetectionCount);
				if (_opt & OPT_TIMER)
				{
					cudaEventRecord(stop_detection);
					cudaEventSynchronize(stop_detection);
					cudaEventElapsedTime(&_timers[TIMER_DETECTION], start_detection, stop_detection);
				}
				_processDetections();
				break;
			}
			case DET_PREFIXSUM:
			{
				if (_opt & OPT_TIMER)
				{
					cudaEventCreate(&start_detection);
					cudaEventCreate(&stop_detection);
					cudaEventRecord(start_detection);
				}
				uint32 sharedMemsize = (_block.x * _block.y * sizeof(SurvivorData)) + (_block.x * _block.y * sizeof(uint32));
				detectionKernel_prefixsum << <grid, _block, sharedMemsize >> >(_finalPyramidTexture, _alphasTexture, _devDetections, _devDetectionCount);
				if (_opt & OPT_TIMER)
				{
					cudaEventRecord(stop_detection);
					cudaEventSynchronize(stop_detection);
					cudaEventElapsedTime(&_timers[TIMER_DETECTION], start_detection, stop_detection);
				}
				_processDetections();
				break;
			}
			case DET_CPU:
			{
				// copy image from texture to GPU mem and then to CPU mem
				float* pyramidImage;
				dim3 grid(_pyramid.canvasWidth / _block.x + 1, _pyramid.canvasHeight / _block.y + 1, 1);
				cudaMalloc((void**)&pyramidImage, _pyramid.canvasImageSize * sizeof(float));
				copyImageFromTextureObject<<<grid, _block>>>(pyramidImage, _finalPyramidTexture, _pyramid.canvasWidth, _pyramid.canvasHeight);

				// float representation is used inside GPU
				// we need to convert it to uint8
				cv::Mat tmp(cv::Size(_pyramid.canvasWidth, _pyramid.canvasHeight), CV_32FC1);	
				uint8* bw = (uint8*)malloc(_pyramid.canvasWidth * _pyramid.canvasHeight);
				
				cudaMemcpy(tmp.data, pyramidImage, _pyramid.canvasImageSize * sizeof(float), cudaMemcpyDeviceToHost);					
				for (int i = 0; i < tmp.rows; i++)
					for (int j = 0; j < tmp.cols; j++)						
						bw[i * tmp.cols + j] = static_cast<uint8>(tmp.at<float>(i, j) * 255.f);				
				cudaFree(pyramidImage);

				ClockPoint det_time_start, det_time_end;
				if (_opt & OPT_TIMER)
					det_time_start = Clock::now();

				std::vector<Detection> detections;
				simple::detect(detections, bw, _pyramid.canvasWidth, _pyramid.canvasHeight, _pyramid.canvasWidth);
				::free(bw);

				if (_opt & OPT_TIMER)
				{
					det_time_end = Clock::now();
					FPDuration duration = det_time_end - det_time_start;
					_timers[TIMER_DETECTION] += static_cast<float>(std::chrono::duration_cast<Nanoseconds>(duration).count()) / 1000000.f;
				}
				_processDetections(&(detections[0]), detections.size());
			}
			break;
		}														

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

			if (_opt & OPT_OUTPUT_CSV)
			{				
				std::ofstream file;
				file.open(_outputFilename, std::ios::out|std::ios::app);
				file << _frame << ";" << _timers[TIMER_PREPROCESS] << ";" << _timers[TIMER_PYRAMID] << ";" << _timers[TIMER_DETECTION] << std::endl;
				file.close();				
			}
		}		
		_frame++;		
	}

	void WaldboostDetector::free()
	{
		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
		{ 
			cudaFree(_devPyramidImage[oct]);
			cudaDestroyTextureObject(_texturePyramidObjects[oct]);
		}

		cudaDestroyTextureObject(_preprocessedImageTexture);
		cudaDestroyTextureObject(_finalPyramidTexture);
		cudaDestroyTextureObject(_alphasTexture);		
			
		cudaFree(_devOriginalImage);
		cudaFree(_devPreprocessedImage);
		cudaFree(_devPyramidImage);
		cudaFree(_devAlphaBuffer);
		cudaFree(_devDetections);
		cudaFree(_devDetectionCount);
		cudaFree(_devSurvivors);
		cudaFree(_devSurvivorCount);
	}
}
