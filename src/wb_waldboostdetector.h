#ifndef H_WB_WALDBOOSTDETECTOR
#define H_WB_WALDBOOSTDETECTOR

#include "wb_structures.h"
#include "general.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

namespace wb {	

	__device__ bool eval(uint32 x, uint32 y, float* response, uint16 startStage, uint16 endStage);

	texture<float,2> textureWorkingImage;
	texture<float,2> textureImagePyramid;
	texture<float> textureAlphas;

	__constant__ ImageInfo devInfo[1];
	#define DEV_INFO devInfo[0]

	__constant__ Pyramid devPyramid[1];
	#define PYRAMID devPyramid[0]

	__constant__ Stage stages[STAGE_COUNT];

	class WaldboostDetector 
	{
		public:
			void init(cv::Mat* image);
			void setImage(cv::Mat* image);
			void run();
			void free();

		protected:
			void buildPyramid();
			void bilinearInterpolation();			

		private:
			ImageInfo _info;
			Pyramid _pyramid;
			uint8* _devOriginalImage;			
			float* _devWorkingImage;			
			float* _devPyramidImage;
			float* _devAlphaBuffer;
			Detection* _devDetections;
			uint32* _devDetectionCount;
			SurvivorData* _devSurvivors;

			#ifdef DEBUG
			float _totalTime;
			uint32 _frameCount;
			#endif

				
	};
}

#endif
