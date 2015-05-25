#include "wbd_simple.h"
#include "wbd_alphas.h"
#include "wbd_structures.h"
#include "wbd_detector.h"

#include <vector>
#include <iostream>

namespace wbd 
{
	namespace simple 
	{
        namespace pyramid
        {           
            cv::Mat createPyramidImage(cv::Mat const& input, uint8 octaves, uint8 levelsPerOctave)
            {
                float canvasHeight, canvasWidth;

                canvasWidth = input.cols + input.cols / 2.0f;
                float tmp = 0;
                for (uint8 i = 0; i <= levelsPerOctave; ++i)
                    tmp += pow(2, i / levelsPerOctave);
                canvasHeight = tmp * input.rows;

                cv::Mat grayscale;
                cv::cvtColor(input, grayscale, CV_BGR2GRAY);

                cv::Mat pyramid(static_cast<uint32>(canvasHeight), static_cast<uint32>(canvasWidth), CV_8UC1);
                float scalingFactor = powf(2, 1.f / static_cast<float>(levelsPerOctave));
                                
                uint32 widthOffset = 0;
                uint32 heightOffset = 0;
                uint32 oldHeightOffset = 0;
                float scale = 1.f;
                for (uint8 i = 0; i < octaves; ++i)
                {                      
                    if (i == 1)
                    {
                        widthOffset += input.cols;
                        heightOffset = 0;
                    }

                    if (i == 3)
                    { 
                        widthOffset += input.cols / 4;
                        heightOffset = oldHeightOffset;
                    }

                    if (i == 2)
                        oldHeightOffset = heightOffset;
                             
                    for (uint8 j = 0; j < levelsPerOctave; ++j)
                    {
                        cv::Mat tmp;                                                
                        cv::resize(grayscale, tmp, cv::Size(grayscale.cols / scale, grayscale.rows / scale));                        
                        tmp.copyTo(pyramid(cv::Rect(widthOffset, heightOffset, tmp.cols, tmp.rows)));
                        scale *= scalingFactor;
                        heightOffset += tmp.rows;
                    }
                }

                return pyramid;
            }                    
        } // pyramid

		void detect(std::vector<Detection>& detections, uint8* image, uint32 const& width, uint32 const& height, uint32 const& pitch)
		{			
			for (uint32 x = 0; x < width - WB_CLASSIFIER_WIDTH; ++x)
			{
				for (uint32 y = 0; y < height - WB_CLASSIFIER_HEIGHT; ++y)
				{					
					float response = 0.f;
					if (eval(image, x, y, pitch, &response))
					{
						Detection d;
						d.width = WB_CLASSIFIER_WIDTH;
						d.height = WB_CLASSIFIER_HEIGHT;
						d.x = x;
						d.y = y;
						d.response = response;

						detections.push_back(d);
					}
				}
			}			
		}

		void sumRegions(uint8* image, uint32 const& x, uint32 const& y, uint32 pitch, uint32* values, Stage* stage)
		{
			image += y * pitch + x;
			uint32 blockStep = stage->height * pitch;
			pitch -= stage->width;			

			uint8 * base[9] = {
				image, image + stage->width, image + 2 * stage->width,
				image + blockStep, image + blockStep + stage->width, image + blockStep + 2 * stage->width,
				image + 2 * blockStep, image + 2 * blockStep + stage->width, image + 2 * blockStep + 2 * stage->width,
			};

			for (uint8 y = 0; y < stage->height; ++y)
			{
				uint8 x = 0;
				while (x < stage->width)
				{
					for (uint8 i = 0; i < 9; ++i)
					{						
						values[i] += *base[i];
						++base[i];
					}
					++x;
				} 
				for (uint8 i = 0; i < 9; ++i)
					base[i] += pitch;
			}
		}

		float evalLBP(uint8* image, uint32 const& x, uint32 const& y, uint32 const& pitch, Stage* stage)
		{
			const uint8 order[] = { 0, 1, 2, 5, 8, 7, 6, 3 };

			uint32 values[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

			sumRegions(image, x + stage->x, y + stage->y, pitch, values, stage);			

			uint8 code = 0;
			for (uint8 i = 0; i < 8; ++i)
				code |= (values[order[i]] > values[4]) << i;			
			
			return alphas[stage->alphaOffset + code];
		}

		bool eval(uint8* image, uint32 const& x, uint32 const& y, uint32 const& pitch, float* response)
		{			
			for (uint16 i = 0; i < WB_STAGE_COUNT; ++i)
			{
				Stage stage = hostStages[i];
				*response += evalLBP(image, x, y, pitch, &stage);
				if (*response < stage.thetaB) {
					return false;
				}
			}			

			return *response > WB_FINAL_THRESHOLD;
		}
	}
}
