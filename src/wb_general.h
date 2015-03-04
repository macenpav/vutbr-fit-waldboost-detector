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
#define SCALING_FACTOR 1.1892071f
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

/** @brief Debug mode */
#define DEBUG

#endif
