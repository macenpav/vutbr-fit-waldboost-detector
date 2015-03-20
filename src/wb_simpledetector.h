/**
* @file wb_simpledetector.h
* @brief Simple c++ waldboost detector.
* 		 
* @details A simple c++ implementation of a waldboost
* 		   mainly for comparison measurements.
*
* @author Pavel Macenauer <macenauer.p@gmail.com>
*/

#ifndef H_WB_SIMPLEDETECTOR
#define H_WB_SIMPLEDETECTOR

#include "wb_structures.h"
#include "wb_general.h"

#include <vector>

namespace wb
{
	namespace simple
	{		
		void detect(
			std::vector<Detection>& detections,
			uint8* image,
			uint32 const& width,
			uint32 const& height,
			uint32 const& pitch
		);

		bool eval(
			uint8*			image, 
			uint32 const&	x, 
			uint32 const&	y, 
			uint32 const&	pitch, 
			float*			response
		);

		/** @brief Evaluates LBP for a given coordinate
		*
		* @details Evaluates LBP for a given coordinate with a given stage and returns a response.
		*
		* @param x				X-coordinate.
		* @param y				Y-coordinate.
		* @param stage			Classifier stage.
		* @return				A response.
		*/
		float evalLBP(
			uint8*			image, 
			uint32 const&	x, 
			uint32 const&	y, 
			uint32 const&	pitch, 
			Stage*			stage
		);

		/** @brief Sums regions for LBP calculation.
		*
		* @details Interpolates image regions (1x1, 2x1, 1x2, 2x2) for LBP calculation. Uses
		* texture unit bilinear interpolation capabilities.
		*
		* @param values	Values used for LBP calculation.
		* @param x			X-coordinate.
		* @param y			Y-coordinate.
		* @param stage		Classifier stage.
		* @return			Void.
		*/
		void sumRegions(
			uint8*			image, 
			uint32 const&	x, 
			uint32 const&	y, 
			uint32			pitch, 
			uint32*			values, 
			Stage*			stage
		);
	}
}

#endif
