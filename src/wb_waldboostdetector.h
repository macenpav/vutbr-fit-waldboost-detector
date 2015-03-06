/**   
 * @file	wb_waldboostdetector.h
 * @brief	Waldboost detector.
 *
 * Global and device functions (all in wb namespace) and a WaldboostDetector class
 * which uses the as gpu kernels for object detection using waldboost metaalgorithm
 * and LBP features. Only the WaldboostDetector class should be used on its own.
 *
 * @author Pavel Macenauer <macenauer.p@gmail.com>
 */

#ifndef H_WB_WALDBOOSTDETECTOR
#define H_WB_WALDBOOSTDETECTOR

#include "wb_structures.h"
#include "wb_general.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

namespace wb {

	/** @brief Initial survivor detection processing
	 *
	 * Processes detections on an image from the first stage (of the waldboost detector).
	 * Processes the whole image and outputs the remaining surviving positions after reaching
	 * the ending stage.
	 *
	 * @param survivors			Output array of surviving positions.
	 * @param endStage			Ending stage of the waldboost detector.
	 * @return					Void.
	 */
	__device__
		void detectSurvivorsInit(SurvivorData* survivors, uint16 endStage);

	/** @brief Survivor detection processing
	 *
	 * Processes detections on an image from a set starting stage (of the waldboost detector).
	 * Processes only positions in the initSurvivors array and outputs still surviving positions
	 * after reaching the ending stage.
	 *
	 * @param survivors			Output and input array of surviving positions.
	 * @param startStage		Starting stage of the waldboost detector.
	 * @param endStage			Ending stage of the waldboost detector.
	 * @return					Void.
	 */
	__device__
		void detectSurvivors(SurvivorData* survivors, uint16 startStage, uint16 endStage);

	/** @brief Final detection processing
	 *
	 * Processes detections on an image beginning at a starting stage, until the end.
	 * Processes only given surviving positions and outputs detections, which can then
	 * be displayed.
	 *
	 * @param survivors			Input array of surviving positions.
	 * @param detections		Output array of detections.
	 * @param detectionCount	Number of detections.
	 * @param startStage		Starting stage of the waldboost detector.
	 * @return					Void.
	 */
	__device__
		void detectDetections(SurvivorData* survivors, Detection* detections, uint32* detectionCount, uint16 startStage);

	/** @brief Evaluates stages for a given coordinate
	 *
	 * Evaluates stages for a given coordinate from a starting stage to an end stage,
	 * accumulates a response and determines if given sample is an object.
	 *
	 * @param x				X-coordinate.
	 * @param y				Y-coordinate.
	 * @param response		Accumulated reponse.
	 * @param startStage	Starting stage.
	 * @param endStage		Ending stage.
	 * @return				Detection success.
	 */
	__device__
		bool eval(uint32 x, uint32 y, float* response, uint16 startStage, uint16 endStage);

	/** @brief Evaluates LBP for a given coordinate
	 *
	 * Evaluates LBP for a given coordinate with a given stage and returns a response.
	 *
	 * @param x				X-coordinate.
	 * @param y				Y-coordinate.
	 * @param stage			Classifier stage.
	 * @return				A response.
	 */
	__device__
		float evalLBP(uint32 x, uint32 y, Stage* stage);

	/** @brief Sums regions for LBP calculation.
	 *
	 * Interpolates image regions (1x1, 2x1, 1x2, 2x2) for LBP calculation. Uses
	 * texture unit bilinear interpolation capabilities.
	 *
	 * @param values	Values used for LBP calculation.
	 * @param x			X-coordinate.
	 * @param y			Y-coordinate.
	 * @param stage		Classifier stage.
	 * @return			Void.
	 */
	__device__
		void sumRegions(float* values, uint32 x, uint32 y, Stage* stage);

	/** @brief Preprocessing kernel.
	 *
	 * GPU kernel doing preprocessing - converts image to black and white and integer
	 * values to float.	FP values are then stored as a texture.
	 *
	 * @param outData	Output data.
	 * @param inData	Input data.
	 * @return			Void.
	 */
	__global__
		void preprocessKernel(float* outData, uint8* inData);

	/** @brief Pyramidal image kernel.
	*
	* GPU kernel, which creates a pyramidal image from a texture. Uses texture unit
	* bilinear interpolation.
	*
	* @param outData	Output data.
	* @return			Void.
	*/
	__global__
		void pyramidKernel(float* outData);

	/** @brief Black and white floating=point texture. */
	texture<float, 2> textureWorkingImage;

	/** @brief Pyramid image texture. */
	texture<float, 2> textureImagePyramid0;
	texture<float, 2> textureImagePyramid1;

	/** @brief Detector alphas saved as texture. */
	texture<float> textureAlphas;

	/** @brief Image information. */
	__constant__ ImageInfo devInfo[1];
	#define DEV_INFO devInfo[0]

	/** @brief Pyramid image information. */
	__constant__ Pyramid devPyramid[1];
	#define PYRAMID devPyramid[0]

	/** @brief Detector stages. */
	__constant__ Stage stages[STAGE_COUNT];

	class WaldboostDetector 
	{
		public:
			/** @brief Initializes the detector.
			 *
			 * Initializes the detector based on given image parameters. It's stuff, 
			 * which is called only once for a video or an image, such as gpu memory
			 * allocation
			 *
			 * @param image		Pointer to an image.
			 * @return			Void.
			 */
			void init(cv::Mat* image);

			/** @brief Sets kernel configuration. 
			 *
			 * Sets the configuration such as how many blocks and threads to use, how
			 * they are organized and so on for easier detector manipulation. 
			 *
			 * @param config	Passed configuration.
			 * @returns			Void.
			 */
			void setAttributes(DetectorConfiguration const& config);

			/** @brief Passes an image to the detector. 
			 *
			 * Passes image to the detector and does the preprocessing. This means, it
			 * feeds the image data to the gpu, converts it to float/black and white and 
			 * generates a pyramid image.
			 *
			 * @param image		Pointer to an image.
			 * @return			Void.
			 */
			void setImage(cv::Mat* image);

			/** @brief Processes detections. 
			 *
			 * Runs the detector, that means processes detections on a pyramid image
			 * saved in texture memory.
			 *
			 * @return Void.
			 */
			void run();

			/** @brief Cleans up memory. 
			 *			
			 * @return Void.
			 */
			void free();	

		private:
			/** @brief Precalculates images sizes, offsets, and so on ... 
			 *
			 * @return Void.
			 */
			void _precalcPyramid();

			ImageInfo _info;						///< image information
			Pyramid _pyramid;						///< pyramid image information

			uint8* _devOriginalImage;				///< pointer to device original image memory
			float* _devWorkingImage;				///< pointer to device preprocessed image memory
			float* _devPyramidImage0, *_devPyramidImage1;	///< pointers to device pyramid image memory
			float* _devAlphaBuffer;					///< pointer to device alpha memory
			Detection* _devDetections;				///< pointer to the detections in device memory
			uint32* _devDetectionCount;				///< pointer to the number of detections in device memory
			SurvivorData* _devSurvivors;			///< pointer to device survivor memory
			DetectorConfiguration _config;			///< detector configuration

			#ifdef WB_DEBUG
			float _totalTime;			///< total time class runs
			uint32 _frameCount;			///< number of frames it processed
			#endif

				
	};
}

#endif
