/**
 * @file wb_simpledetector.h
 * @brief Simple c++ waldboost detector.
 * 		 
 * @details A simple c++ implementation of a waldboost detector
 * 			mainly for comparison measurements.
 *
 * @author Pavel Macenauer <macenauer.p@gmail.com>
 */

#ifndef H_WB_SIMPLEDETECTOR
#define H_WB_SIMPLEDETECTOR

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "wbd_general.h"

namespace wbd
{
	struct Stage;
	struct Detection;

    /// @todo parallelize this implementation with threads
	namespace simple
	{		
        namespace detection
        {
            /** @brief Runs object detection.
             *
             * @details Runs object detector for a given image.
             *
             * @param detections	A vector with detections.
             * @param image			Image data.
             * @param width			Image width.
             * @param height		Image height.
             * @param pitch			Image pitch (row width).
             * @return				A void.
             */
            void detect(
                std::vector<Detection>& detections,
                uint8*					image,
                uint32 const&			width,
                uint32 const&			height,
                uint32 const&			pitch
            );

            /** @brief Classifies an object.
             *
             * @details Goes through all the stages of the classifier, evaluates weak classifiers
             *			and returns a strong classifier.
             *
             * @param image		Image data.
             * @param x			X-coordinate.
             * @param y			Y-coordinate.
             * @param pitch		Image pitch (row width).
             * @param stage		LBP classifier stage.
             * @return			Object found or not.
             */
            bool eval(
                uint8*			image,
                uint32 const&	x,
                uint32 const&	y,
                uint32 const&	pitch,
                float*			response
            );

            /** @brief Evaluates LBP for a given X-Y coordinate.
             *
             * @details Evaluates LBP for a given coordinate and a stage, returning a response.
             *
             * @param image		Image data.
             * @param x			X-coordinate.
             * @param y			Y-coordinate.
             * @param pitch		Image pitch (row width).
             * @param stage		LBP classifier stage.
             * @return			A response.
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
             * @details Sums up image regions based on stage width and height (1x1 to 2x2) mostly.
             * 			This is later on used in LBP code calculation.
             *
             * @param image		Image data.
             * @param x			X-coordinate.
             * @param y			Y-coordinate.
             * @param pitch		Image pitch (row width).
             * @param values	Accumulated values.
             * @param stage		LBP classifier stage.
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
        } // namespace detection

        namespace pyramid
        {
            /** @brief Creates pyramidal image 
             *
             * @details Creates a pyramidal image. The first image is left, second is right, third
             *          is below the second and the fourth is on the right of the third.
             *
             * @param input             Input image (matrix).
             * @param octaves           Number of octaves. The smallest image in an octave is 1/2 of the largest.
             * @param levelsPerOctave   Number of images per octave.
             * @return                  Created pyramid.
             */
            cv::Mat createPyramidImage(
                cv::Mat const&  input, 
                const uint8     octaves, 
                const uint8     levelsPerOctave
            );
        } // namespace pyramid
	}
}

#endif
