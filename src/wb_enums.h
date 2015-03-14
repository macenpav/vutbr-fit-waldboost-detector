/**
* @file	wb_enums.h
* @brief	Waldboost detector enums.
*
* @details All enums are held hear. That means run parameters, pyramid generation modes and
* other such options. When implementing a way how to generate a pyramid a new mode
* should be defined here.
*
* @author Pavel Macenauer <macenauer.p@gmail.com>
*/

#ifndef H_WB_ENUMS
#define H_WB_ENUMS

namespace wb {

	/** @brief Allowed input types for a sample program */
	enum InputTypes {
		/** @brief List of images in a textfile */
		INPUT_IMAGE_DATASET = 1,
		/** @brief Video */
		INPUT_VIDEO
	};

	/** @brief Run option parameters */
	enum RunOptions {
		/** @brief Displays final image with detections. */
		OPT_VISUAL_OUTPUT	= 0x00000001,
		/** @brief Outputs more info than usual. */
		OPT_VERBOSE			= 0x00000002,
		/** @brief Time is measured for the pyramid, detection, preprocessing and so on. */
		OPT_TIMER			= 0x00000004,
		/** @brief Outputs different stages of the detector. */
		OPT_VISUAL_DEBUG	= 0x00000008,
		/** @brief Output measurements to a CSV file. */
		OPT_OUTPUT_CSV		= 0x00000010|OPT_TIMER,
		
		/** @brief Max. number of frames processed limited. */
		OPT_LIMIT_FRAMES	= 0x00000020,

		/** @brief All options are switched on. */
		OPT_ALL				= 0xFFFFFFFF
	};

	/** @brief How the pyramid is generated. */
	enum PyramidGenModes {
		/** @brief Pyramid generated inside a texture from the same texture. */
		PYGEN_SINGLE_TEXTURE,
		/** @brief Textures generated separately and the final image is then copied. */
		PYGEN_BINDLESS_TEXTURE,

		MAX_PYGEN_MODES
	};

	/** @brief How the pyramid image will look like. */
	enum PyramidTypes {
		/** @brief Pyramids positioned in a programmer-defined way */
		PYTYPE_OPTIMIZED,
		/** @brief Pyramids positioned next to each other */
		PYTYPE_HORIZONAL,

		MAX_PYTYPES
	};

	/** @brief Timers for separate measurements. */
	enum Timers {
		/** @brief Pyramid generation timer. */
		TIMER_PYRAMID,
		/** @brief Object detection timer. */
		TIMER_DETECTION,
		/** @brief Image preprocessing timer. */
		TIMER_PREPROCESS,
		/** @brief Init timer. */
		TIMER_INIT,
		/** @brief Total time. */
		TIMER_TOTAL,

		MAX_TIMERS
	};
}

#endif
