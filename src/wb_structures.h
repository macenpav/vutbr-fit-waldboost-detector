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

	struct ImageData {
		uint32 width, height;
		uint32 offset, offsetX, offsetY;
		float scale;
	};

	struct Octave {
		ImageData images[WB_LEVELS_PER_OCTAVE];
	};

	struct Pyramid {
		uint32 width[WB_OCTAVES], height[WB_OCTAVES], imageSize[WB_OCTAVES];
		uint32 canvasWidth[WB_OCTAVES], canvasHeight[WB_OCTAVES], canvasImageSize[WB_OCTAVES];
		Octave octaves[WB_OCTAVES];
	};

	struct SurvivorData {
		uint32 x, y;
		float response;
	};

	struct DetectorConfiguration {
		dim3 kernelConfig[MAX_KERNEL_CONFIG];
	};		
}

#endif
