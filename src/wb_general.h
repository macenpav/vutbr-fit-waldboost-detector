/**
* @file	wb_general.h
* @brief Waldboost detector general settings.
*
* Settings and constants are held here.
*
* @author Pavel Macenauer <macenauer.p@gmail.com>
*/

#ifndef H_GENERAL
#define H_GENERAL

#include <string>
#include <chrono>

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

typedef char int8;
typedef short int16;
typedef int int32;

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds Milliseconds;
typedef std::chrono::duration<float> FPDuration;
typedef std::chrono::system_clock::time_point ClockPoint;

/** @brief Final detector threshold. */
#define FINAL_THRESHOLD 0.f

/** @brief Number of alphas in the detector. */
#define ALPHA_COUNT 256

/** @brief Number of stages of the detector. */
#define STAGE_COUNT 2048

/** @brief GPU block size (used as shared memory size). */
#define BLOCK_SIZE 1024

/** @brief Discarded sample. */
#define WB_BAD_RESPONSE -5000.f

/** @brief Delay in ms between displaying frames */
#define WB_WAIT_DELAY 1

/** @brief Max. number of detections per frame. 
 *
 * Allocates memory and cannot overflow, but can be kept lower to save GPU memory.
 */
#define WB_MAX_DETECTIONS 512

/** @brief Classifier window width. */
#define WB_CLASSIFIER_WIDTH 26

/** @brief Classifier window height. */
#define WB_CLASSIFIER_HEIGHT 26

/** @brief Downsampling scaling factor. 
 *
 * @details 1.09050773f = 2^(1/8)
 * Should be 2^(1/WB_LEVELS_PER_OCTAVE), because the number of levels says
 * how many downsampled images are created before an image of 1/2 width/height
 * is created. Smaller number would generate bigger images, bigger number
 * smaller images and the last image wouldn't be half the width/height. 
 */
#define WB_SCALING_FACTOR 1.09050773f

/** @brief Number of downsampled images before the original image is 1/2 the width/height */
#define WB_LEVELS_PER_OCTAVE 8

/** @brief Number of image sets sampled from 1:1 to 1:0.5 scale */
#define WB_OCTAVES 4

/** @brief Library name. */
const std::string LIBNAME = "waldboost-detector";

/** @brief Just a helper so we don't have to write the whole thing in debug output. */
const std::string LIBHEADER = "[" + LIBNAME + "]: ";

#endif
