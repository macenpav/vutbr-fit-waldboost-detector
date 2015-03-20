/**   
 * @file wb_waldboostdetector.h
 * @brief Waldboost detector.
 *
 * @details Global and device functions (all in wb namespace) and a WaldboostDetector class
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
	 * @details Processes detections on an image from the first stage (of the waldboost detector).
	 * Processes the whole image and outputs the remaining surviving positions after reaching
	 * the ending stage.
	 *
	 * @param survivors			Output array of surviving positions.
	 * @param endStage			Ending stage of the waldboost detector.
	 * @return					Void.
	 */
	__device__
	void detectSurvivorsInit_prefixsum(
		uint32 const&	x,
		uint32 const&	y,
		uint32 const&	threadId,
		uint32 const&	globalOffset,
		uint32 const&	blockSize,
		SurvivorData*	survivors,
		uint32&			survivorCount,
		uint32*			survivorScanArray,
		uint16			endStage
	);

	/** @brief Survivor detection processing
	 *
	 * @details Processes detections on an image from a set starting stage (of the waldboost detector).
	 * Processes only positions in the initSurvivors array and outputs still surviving positions
	 * after reaching the ending stage.
	 *
	 * @param survivors			Output and input array of surviving positions.
	 * @param startStage		Starting stage of the waldboost detector.
	 * @param endStage			Ending stage of the waldboost detector.
	 * @return					Void.
	 */
	__device__
	void detectSurvivors_prefixsum(
		uint32 const&		threadId,
		uint32 const&		globalOffset,
		uint32 const&		blockSize,
		SurvivorData*		survivors,
		uint32&				survivorCount,
		uint32*				survivorScanArray,
		uint16				startStage,
		uint16				endStage
	);

	/** @brief Final detection processing
	 *
	 * @details Processes detections on an image beginning at a starting stage, until the end.
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
	void detectDetections_prefixsum(
		uint32 const& threadId, 
		uint32 const&	globalOffset,
		SurvivorData* survivors, 
		Detection* detections, 
		uint32* detectionCount, 
		uint16 startStage
	);

	/** @brief Evaluates stages for a given coordinate
	 *
	 * @details Evaluates stages for a given coordinate from a starting stage to an end stage,
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
	 * @details Evaluates LBP for a given coordinate with a given stage and returns a response.
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
	 * @details Interpolates image regions (1x1, 2x1, 1x2, 2x2) for LBP calculation. Uses
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
	 * @details GPU kernel doing preprocessing - converts image to black and white and integer
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
	* @details GPU kernel, which creates a pyramidal image from a texture. Uses texture unit
	* bilinear interpolation.
	*
	* @param outData	Output data.
	* @return			Void.
	*/
	__global__
	void pyramidKernel(float* outData);

	__global__ 
	void firstPyramidKernel(float* outData, float* finalData);

	__global__ 
	void pyramidFromPyramidKernel(float* outData, float* finalData, cudaTextureObject_t inData, uint8 level);

	/** @brief Copies image from a statically set texture. */
	__global__
	void copyKernel(float* out, uint32 width, uint32 height);

	/** @brief Copies image from a dynamically set texture. */
	__global__
	void copyKernel(float* out, cudaTextureObject_t obj, uint32 width, uint32 height);

	/** @brief Clears an image.
	 * 
	 * Sets all floating-point pixels to 0.
	 */
	__global__
	void clearKernel(float* data, uint32 width, uint32 height);

	/** @brief Black and white floating-point texture. */
	texture<float, 2> textureWorkingImage;

	/** @brief Final texture used for detection */
	texture<float, 2> texturePyramidImage;

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
			 * @details Initializes the detector based on given image parameters. It's stuff, 
			 * which is called only once for a video or an image, such as gpu memory
			 * allocation
			 *
			 * @param image		Pointer to an image.
			 * @return			Void.
			 */
			void init(cv::Mat* image);

			/** @brief Passes an image to the detector. 
			 *
			 * @details Passes image to the detector and does the preprocessing. This means, it
			 * feeds the image data to the gpu, converts it to float/black and white and 
			 * generates a pyramid image.
			 *
			 * @param image		Pointer to an image.
			 * @return			Void.
			 */
			void setImage(cv::Mat* image);

			/** @brief Processes detections. 
			 *
			 * @details Runs the detector, that means processes detections on a pyramid image
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

			/** @brief Sets pyramid generation mode. */
			void setPyGenMode(PyramidGenModes const& mode) { _pyGenMode = mode; } 

			/** @brief Sets pyramid type. */
			void setPyType(PyramidTypes const& type) { _pyType = type; }

			/** @brief Sets detection mode. */
			void setDetectionMode(DetectionModes const& mode) { _detectionMode = mode; }

			/** @brief Sets the kernel block size. */
			void setBlockSize(uint32 const& x = 32, uint32 const& y = 32, uint32 const& z = 1){ _block = dim3(x, y, z); }

			/** @brief Passes run parameters to the detector. */
			void setRunOptions(uint32 const& options) { _opt = options; }

			/** @brief Sets a file for output. */
			void setOutputFile(std::string const& output) { _outputFilename = output; }			

			uint32 getFrameCount() const { return _frame; }

		private:
			/** @brief Precalculates image sizes and offsets horizontally.
			 *
			 * @details Precalculates pyramids in a horizontal manner, that means every pyramid next to each other.
			 *
			 * @return Void.
			 * @todo fix crash
			 */
			void _precalcHorizontalPyramid();

			/** @brief Precalculates image sizes and offsets in a user defined manner.
			 *
			 * @details Precalculates pyramids in a user defined manner. Currently 1st octave is on the left, 2nd
			 * top-right, 3rd bottom-left (right from 1st), 4th bottom-right.
			 *
			 * @return Void.
			 */
			void _precalc4x8Pyramid();

			/** @brief Wrapper around pyramid kernels. 
			 *
			 * @return Void. 
			 */
			void _pyramidKernelWrapper();

			/** @brief Single texture pyramid generation.
			 *
			 * @details Pyramid is generated by creating a single texture canvas, generating the first octave from 
			 * the original preprocessed image and then creating downsapled pyramids from the same texture, 
			 * which leads reads and writes from the same texture. Every octave needs only its width x height
			 * number of threads, that means, that every octave several threads (and blocks) return.
			 *
			 * @return Void.
			 */
			void _pyramidGenSingleTexture();

			/** @brief Bindless texture pyramid generation. 
			 *
			 * @details Pyramid is generated using multiple textures, always downsampling from a previously generated textury.
			 * This leads to reads only from textures. While a new texture is generated, the result is simultaneously
			 * copied to the final texture. Disadvantage of this method is initialization for every texture and
			 * kernel run per octave.
			 *
			 * @return Void.
			 */
			void _pyramidGenBindlessTexture();
			
			/** @brief Clears timers.
			 *
			 * @return Void.
			 */
			void _initTimers();

			/** @brief Recalcs detections. 
			 *
			 * @details Detections are detected on an image containing lots of downsampled images (pyramids). Here we map detected
			 * positions to the original image.
			 *
			 * @return Void.
			 */
			void _processDetections();


			ImageInfo			_info;				///< image information

			Pyramid				_pyramid;			///< pyramid image information
			PyramidGenModes		_pyGenMode;			///< pyramid generation mode
			PyramidTypes		_pyType;			///< pyramid type (look)
			DetectionModes		_detectionMode;		///< detection mode
			uint32				_opt;				///< run options/parameters
			float				_timers[MAX_TIMERS];///< timers
			dim3				_block;				///< kernel block size

			uint8*				_devOriginalImage;					///< pointer to device original image memory
			float*				_devWorkingImage;					///< pointer to device preprocessed image memory					

			float*				_devPyramidData;					///< pointer to device pyramid memory (used by single texture)

			float*				_devPyramidImage[WB_OCTAVES];		///< pointer to device pyramid memory (used by bindless texture)	
			cudaTextureObject_t	_texturePyramidObjects[WB_OCTAVES]; ///< cuda texture objects (used by bindless texture)

			float*				_devAlphaBuffer;		///< pointer to device alpha memory
			Detection*			_devDetections;			///< pointer to the detections in device memory
			uint32*				_devDetectionCount;		///< pointer to the number of detections in device memory
			SurvivorData*		_devSurvivors;			///< pointer to device survivor memory			
			uint32*				_devSurvivorCount;
			cv::Mat*			_myImage;				///< pointer to the original processed image
			std::string			_outputFilename;		///< filename for csv output

			uint32				_frame;					///< frame counter
	};
}

#endif
