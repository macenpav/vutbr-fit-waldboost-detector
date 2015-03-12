#ifndef H_WB_STRUCTURES
#define H_WB_STRUCTURES

#include "wb_general.h"
#include "wb_enums.h"

namespace wb {

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

	struct ImageInfo {
		uint32	width, height, imageSize;
		uint8	channels;
	};

	struct PyramidImage {
		uint32 width, height;
		uint32 offsetX, offsetY;
		float scale;
	};

	struct Octave {
		uint32 width, height, imageSize;
		uint32 offsetX, offsetY;
		PyramidImage images[WB_LEVELS_PER_OCTAVE];
	};

	struct Pyramid {
		uint32 canvasWidth, canvasHeight, canvasImageSize;		
		Octave octaves[WB_OCTAVES];
	};

	struct SurvivorData {
		uint32 x, y;
		float response;
	};
}

#endif
