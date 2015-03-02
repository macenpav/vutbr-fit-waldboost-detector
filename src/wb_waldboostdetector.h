#ifndef H_WB_WALDBOOSTDETECTOR
#define H_WB_WALDBOOSTDETECTOR

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "cuda_runtime.h"

#include "general.h"

namespace wb {

	__global__ void kernel(float* imageData);	
	
	struct ImageInfo {
		uint32 width, height, imageSize;
		uint8 channels;
	};	

	__constant__ ImageInfo devInfo[1];
	#define DEV_INFO devInfo[0]

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
			uint8* _devOriginalImage;			
			float* _devWorkingImage;
	};
}

#endif
