#ifndef H_WB_ENUMS
#define H_WB_ENUMS

namespace wb {

	/** @brief Allowed input types for a sample program */
	enum WBInputTypes {
		/** @brief List of images in a textfile */
		INPUT_IMAGE_DATASET = 1,
		/** @brief Video */
		INPUT_VIDEO
	};

	/** @brief Run option parameters */
	enum WBRunOptions {
		/** @brief show visual output using opencv */
		OPT_VISUAL_OUTPUT	= 0x00000001,
		/** @brief output all debugging info */
		OPT_VERBOSE			= 0x00000002,
		OPT_ALL				= 0xFFFFFFFF
	};
}

#endif
