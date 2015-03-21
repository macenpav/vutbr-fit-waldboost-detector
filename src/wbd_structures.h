/**
 * @file	wb_enums.h
 * @brief Waldboost detector structures.
 *
 * Used structures are held here.
 *
 * @author Pavel Macenauer <macenauer.p@gmail.com>
 */

#ifndef H_WB_STRUCTURES
#define H_WB_STRUCTURES

#include "wbd_general.h"
#include "wbd_enums.h"

namespace wbd 
{

	/** @brief A waldboost detection stage */
	struct Stage {
		/** @brief position */
		uint8 x, y;
		/** @brief size */
		uint8 width, height;
		/** @brief compared response */
		float thetaB;
		/** @brief alpha offset in alpha array */
		uint32 alphaOffset;
	};

	/** @brief Allowed input types for a sample program */
	struct Detection {
		/** @brief position */
		uint32 x, y;
		/** @brief size */
		uint32 width, height;
		/** @brief response */
		float response;
	};

	/** @brief Info about an image. 
	 *
	 * Info about the image, which is passed to the constant memory of the GPU.
	 */
	struct ImageInfo {
		uint32	width, height, imageSize;
		uint8	channels;
	};

	/** @brief Info about an image inside a pyramid. */
	struct PyramidImage {
		uint32 width, height;
		uint32 offsetX, offsetY;
		float scale;
	};

	/** @brief Info about an octave. 
	 *
	 * An octave is a set of downsampled images, where the smallest is 1/2 the size
	 * of the biggest.
	 */
	struct Octave {
		uint32 width, height, imageSize;
		uint32 offsetX, offsetY;
		PyramidImage images[WB_LEVELS_PER_OCTAVE];
	};

	/** @brief Info about a pyramid.
	 *
	 * A pyramid is a set of octaves, where the biggest image of an octave is always 1/2
	 * the size of the biggest image inside the previous octave.
	 */
	struct Pyramid {
		uint32 canvasWidth, canvasHeight, canvasImageSize;		
		Octave octaves[WB_OCTAVES];
	};

	/** @brief Info about a still running detection. 
	 *
	 * Threads calculating detections get discarded, but those still running keep their
	 * data in this structure, so we can continue with computation, when we rerun detection.
	 */
	struct SurvivorData {
		uint32 x, y;
		float response;
	};

	/** @brief Functional detector settings. 
	 *
	 * Structure to hold settings, which change the behaviour of the detector. Default parameters
	 * are also set here. Should be the most effective settings.
	 */
	struct RunSettings
	{
		std::string outputFilename;
		uint32 blockSize				= 32;
		PyramidGenModes pyGenMode		= PYGEN_BINDLESS_TEXTURE;
		PyramidTypes pyType				= PYTYPE_OPTIMIZED;
		DetectionModes detectionMode	= DET_ATOMIC_GLOBAL;
		uint32 maxFrames				= 0;
	};
}

#endif
