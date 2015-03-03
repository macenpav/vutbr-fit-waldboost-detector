#ifndef H_WB_STRUCTURES
#define H_WB_STRUCTURES

#include "general.h"

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

	struct Pyramid {
		uint32	width, height, imageSize;
		uint32	yOffsets[PYRAMID_IMAGE_COUNT];
		float	scales[PYRAMID_IMAGE_COUNT];
		uint32	imageWidths[PYRAMID_IMAGE_COUNT];
		uint32	imageHeights[PYRAMID_IMAGE_COUNT];
	};

	struct SurvivorData {
		uint32 x, y;
		float response;
	};
	
}

#endif
