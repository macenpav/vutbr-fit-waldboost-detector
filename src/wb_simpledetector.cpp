#include "wb_simpledetector.h"
#include "wb_alphas.h"
#include "wb_structures.h"
#include "wb_detector.h"

#include <vector>
#include <iostream>

namespace wb 
{
	namespace simple 
	{
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
			// Prepare pointer array
			uint8 * base[9] = {
				image, image + stage->width, image + 2 * stage->width,
				image + blockStep, image + blockStep + stage->width, image + blockStep + 2 * stage->width,
				image + 2 * blockStep, image + 2 * blockStep + stage->width, image + 2 * blockStep + 2 * stage->width,
			};

			for (uint8 y = 0; y < stage->height; ++y)
			{
				// go through all pixels in row and accumulate
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
				// set pointers to next line 
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
			uint16 i = 0;
			for (; i < WB_STAGE_COUNT; ++i)
			{
				Stage stage = hostStages[i];
				*response += evalLBP(image, x, y, pitch, &stage);
				if (*response < stage.thetaB) {
					return false;
				}
			}			

			// final waldboost threshold
			return *response > WB_FINAL_THRESHOLD;
		}
	}
}
