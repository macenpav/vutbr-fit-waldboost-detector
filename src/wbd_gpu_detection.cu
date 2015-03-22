#include "wbd_gpu_detection.cuh"
#include "wbd_detector.h"

namespace wbd
{
	namespace gpu
	{
		namespace detection
		{
			void initDetectionStages()
			{
				cudaMemcpyToSymbol(stages, hostStages, sizeof(Stage) * WB_STAGE_COUNT);
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
			} // sumRegions

			__device__ float evalLBP(cudaTextureObject_t texture, cudaTextureObject_t alphas, uint32 x, uint32 y, Stage* stage)
			{
				const uint8 LBPOrder[8] = { 0, 1, 2, 5, 8, 7, 6, 3 };

				float values[9];

				sumRegions(texture, values, static_cast<float>(x)+(static_cast<float>(stage->width) * 0.5f), y + (static_cast<float>(stage->height) * 0.5f), stage);

				uint8 code = 0;
				for (uint8 i = 0; i < 8; ++i)
					code |= (values[LBPOrder[i]] > values[4]) << i;

				return tex1Dfetch<float>(alphas, stage->alphaOffset + code);
			} // evalLBP

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
			} // eval

			namespace prefixsum
			{
				__device__ void detectSurvivorsInit
				(
					cudaTextureObject_t texture,
					cudaTextureObject_t alphas,
					uint32 const&		x,
					uint32 const&		y,
					uint32 const&		threadId,
					uint32 const&		globalOffset,
					uint32 const&		blockSize,
					SurvivorData*		survivors,
					uint32&				survivorCount,
					uint32*				survivorScanArray,
					uint16				endStage
				){

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

				__device__ void detectSurvivors
				(
					cudaTextureObject_t texture,
					cudaTextureObject_t alphas,
					uint32 const&		threadId,
					uint32 const&		globalOffset,
					uint32 const&		blockSize,
					SurvivorData*		survivors,
					uint32&				survivorCount,
					uint32*				survivorScanArray,
					uint16				startStage,
					uint16				endStage
				){
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

				__device__ void detectDetections
				(
					cudaTextureObject_t texture,
					cudaTextureObject_t alphas,
					uint32 const&		threadId,
					uint32 const&		globalOffset,
					SurvivorData*		survivors,
					Detection*			detections,
					uint32*				detectionCount,
					uint16				startStage
				){
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

				__global__ void detect(
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

					detectSurvivorsInit(texture, alphas, x, y, threadId, blockOffset, blockSize, survivors, survivorCount, survivorScanArray, 1);

					__syncthreads();
					if (threadId >= survivorCount)
						return;
					__syncthreads();
					if (threadId == 0)
						survivorCount = 0;
					__syncthreads();

					atomicshared::detectSurvivors(texture, alphas, threadId, survivors, &survivorCount, 1, 8);

					__syncthreads();
					if (threadId >= survivorCount)
						return;
					__syncthreads();
					if (threadId == 0)
						survivorCount = 0;
					__syncthreads();
					atomicshared::detectSurvivors(texture, alphas, threadId, survivors, &survivorCount, 8, 64);


					__syncthreads();
					if (threadId >= survivorCount)
						return;
					__syncthreads();
					if (threadId == 0)
						survivorCount = 0;
					__syncthreads();

					atomicshared::detectSurvivors(texture, alphas, threadId, survivors, &survivorCount, 64, 256);

					__syncthreads();
					if (threadId >= survivorCount)
						return;
					__syncthreads();
					if (threadId == 0)
						survivorCount = 0;
					__syncthreads();

					atomicshared::detectSurvivors(texture, alphas, threadId, survivors, &survivorCount, 256, 512);

					__syncthreads();
					if (threadId >= survivorCount)
						return;

					atomicshared::detectDetections(texture, alphas, threadId, survivors, detections, detectionCount, 512);

				}

			} // namespace prefixsum

			namespace atomicshared
			{
				__device__
					void detectSurvivorsInit(
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

				__device__ void detectSurvivors(
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

				__device__
					void detectDetections(
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

				__global__ void detect(
					cudaTextureObject_t texture,
					cudaTextureObject_t alphas,
					uint32				width,
					uint32				height,
					Detection*			detections,
					uint32*				detectionCount)
				{
					extern __shared__ SurvivorData survivors[];
					__shared__ uint32 survivorCount;

					const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
					const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

					if (x < width - WB_CLASSIFIER_WIDTH && y < height - WB_CLASSIFIER_HEIGHT)
					{
						const uint32 threadId = threadIdx.y * blockDim.x + threadIdx.x;

						if (threadId == 0)
							survivorCount = 0;
						__syncthreads();

						detectSurvivorsInit(texture, alphas, x, y, threadId, survivors, &survivorCount, 1);

						__syncthreads();
						if (threadId >= survivorCount)
							return;
						__syncthreads();
						if (threadId == 0)
							survivorCount = 0;
						__syncthreads();

						detectSurvivors(texture, alphas, threadId, survivors, &survivorCount, 1, 8);

						__syncthreads();
						if (threadId >= survivorCount)
							return;
						__syncthreads();
						if (threadId == 0)
							survivorCount = 0;
						__syncthreads();

						detectSurvivors(texture, alphas, threadId, survivors, &survivorCount, 8, 64);

						__syncthreads();
						if (threadId >= survivorCount)
							return;
						__syncthreads();
						if (threadId == 0)
							survivorCount = 0;
						__syncthreads();

						detectSurvivors(texture, alphas, threadId, survivors, &survivorCount, 64, 256);

						__syncthreads();
						if (threadId >= survivorCount)
							return;
						__syncthreads();
						if (threadId == 0)
							survivorCount = 0;
						__syncthreads();

						detectSurvivors(texture, alphas, threadId, survivors, &survivorCount, 256, 512);

						__syncthreads();
						if (threadId >= survivorCount)
							return;
						__syncthreads();

						detectDetections(texture, alphas, threadId, survivors, detections, detectionCount, 512);
					}
				}

			} // namespace atomicshared

			namespace atomicglobal
			{

				__device__
					void detectSurvivorsInit(
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

				__device__ void detectSurvivors(
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

				__device__
					void detectDetections(
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

				__global__ void detect(
					cudaTextureObject_t texture,
					cudaTextureObject_t alphas,
					const uint32		width,
					const uint32		height,
					SurvivorData*		survivors,
					Detection*			detections,
					uint32*				detectionCount)
				{
					const uint32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
					const uint32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

					if (x < width - WB_CLASSIFIER_WIDTH && y < height - WB_CLASSIFIER_HEIGHT)
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

						detectSurvivorsInit(texture, alphas, x, y, threadId, blockOffset, survivors, &blockSurvivors, 1);

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

						detectSurvivors(texture, alphas, threadId, blockOffset, survivors, &blockSurvivors, 1, 8);

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

						detectSurvivors(texture, alphas, threadId, blockOffset, survivors, &blockSurvivors, 8, 64);

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

						detectSurvivors(texture, alphas, threadId, blockOffset, survivors, &blockSurvivors, 64, 256);

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

						detectSurvivors(texture, alphas, threadId, blockOffset, survivors, &blockSurvivors, 256, 512);

						// finish all the detections within a block
						__syncthreads();
						if (threadId >= blockSurvivors)
							return;

						detectDetections(texture, alphas, threadId, blockOffset, survivors, detections, detectionCount, 512);
					}
				}

			} // namespace atomicglobal

		} // namespace detection
	} // namespace gpu
} // namespace wbd
