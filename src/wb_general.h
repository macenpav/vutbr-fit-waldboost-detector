#ifndef H_GENERAL
#define H_GENERAL

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

typedef char int8;
typedef short int16;
typedef int int32;

/** @brief Delay in ms between displaying frames */
#define WAIT_DELAY 1

#define MAX_PYRAMID_WIDTH 320
#define PYRAMID_IMAGE_COUNT 8
#define FINAL_THRESHOLD 0.0f
#define ALPHA_COUNT 256
#define STAGE_COUNT 2048
#define CLASSIFIER_WIDTH 26
#define CLASSIFIER_HEIGHT 26
#define BLOCK_SIZE 1024
#define BAD_RESPONSE -5000.f
#define MAX_DETECTIONS 2048

/** @brief Downsampling scaling factor. 
 *
 * 1.1892071f = 2^(1/4)
 * Should be 2^(1/WB_LEVELS_PER_OCTAVE), because the number of levels says
 * how many downsampled images are created before an image of 1/2 width/height
 * is created. Smaller number would generate bigger images, bigger number
 * smaller images and the last image wouldn't be half the width/height. 
 */
#define WB_SCALING_FACTOR 1.1892071f

/** @brief Number of downsampled images before the original image is 1/2 the width/height */
#define WB_LEVELS_PER_OCTAVE 4

/** @brief Number of image sets sampled from 1:1 to 1:0.5 scale */
#define WB_OCTAVES 2

/** @brief Comment to switch off WB_DEBUG mode */
#define WB_DEBUG

#endif
