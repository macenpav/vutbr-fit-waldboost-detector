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
}

#endif
