#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <fstream>
#include <string>
#include <iostream>
#include <chrono>

#include "wbd_waldboostdetector.cuh"
#include "wbd_enums.h"
#include "wbd_general.h"
#include "wbd_structures.h"
#include "wbd_func.h"

/**
 * @brief Processes an image dataset - a text file with a list of images.
 *
 * @param filename	filename
 * @param opts		run optseters
 */
bool processImageDataset(std::string filename, wbd::RunSettings const& settings, uint32 opts)
{	
	std::ifstream in;
	in.open(filename);

	std::string buffer;
	cv::Mat image;
	wbd::WaldboostDetector detector;    
    detector.setBlockSize(wbd::KERTYPE_PREPROCESS, settings.blockSize, settings.blockSize);
    detector.setBlockSize(wbd::KERTYPE_PYRAMID, settings.blockSize, settings.blockSize);
    detector.setBlockSize(wbd::KERTYPE_DETECTION, settings.blockSize, settings.blockSize);
    detector.setRunOptions(opts);
    detector.setSettings(settings);    

	while (!in.eof())
	{
		std::getline(in, buffer);
        image = cv::imread(buffer.c_str(), CV_LOAD_IMAGE_COLOR);

		if (!image.data)
		{
			std::cerr << LIBHEADER << "Could not open or find the image (filename: " << filename.c_str() << ")" << std::endl;
			continue;
		}

		detector.init(&image);
		if (opts & wbd::OPT_VERBOSE)
			std::cout << LIBHEADER << "Initialized detector." << std::endl;

		detector.setImage(&image);
		if (opts & wbd::OPT_VERBOSE)
			std::cout << LIBHEADER << "Image set." << std::endl;

		if (opts & wbd::OPT_VERBOSE)
			std::cout << LIBHEADER << "Running detections ..." << std::endl;
		detector.run();
		if (opts & wbd::OPT_VERBOSE)
			std::cout << LIBHEADER << "Detection finished." << std::endl;
        
        if (opts & (wbd::OPT_VISUAL_OUTPUT | wbd::OPT_VISUAL_DEBUG))
        {
            cv::imshow(LIBNAME, image);
            cv::waitKey(WB_WAIT_DELAY);
        }

		detector.free();
		if (opts & wbd::OPT_VERBOSE)
			std::cout << LIBHEADER << "Memory free." << std::endl;		
	}

	return true;
}

/**
 * @brief Processes a video.
 *
 * @param filename	filename
 * @param opts		run optseters
 * @todo implement pyType
 */
bool processVideo(std::string const& filename, wbd::RunSettings const& settings, uint32 const& opts)
{
	cv::VideoCapture video;
	cv::Mat image;
	video.open(filename);

	// use first image just to init detector
	video >> image;
	if (image.empty())
		return false;

	wbd::WaldboostDetector detector;	
	detector.setBlockSize(wbd::KERTYPE_PREPROCESS, settings.blockSize, settings.blockSize);
	detector.setBlockSize(wbd::KERTYPE_PYRAMID, settings.blockSize, settings.blockSize);
	detector.setBlockSize(wbd::KERTYPE_DETECTION, settings.blockSize, settings.blockSize);
	detector.setRunOptions(opts);
    detector.setSettings(settings);
	
	detector.init(&image);

	if (opts & wbd::OPT_VERBOSE)	
		std::cout << LIBHEADER << "Initialized detector." << std::endl;			

	while (true)
	{		
		if (opts & wbd::OPT_LIMIT_FRAMES)
		{
			if (settings.maxFrames < detector.getFrameCount())
				break;
		}
				
		if (opts & wbd::OPT_VERBOSE)		
			std::cout << LIBHEADER << "Loading image frame ..." << std::endl;					

		video >> image;		

		if (image.empty())
			break;

		// load detector with an image
		detector.setImage(&image);

		if (opts & wbd::OPT_VERBOSE)
		{ 
			std::cout << LIBHEADER << "Image set." << std::endl;
			std::cout << LIBHEADER << "Running detections ..." << std::endl;
		}

		// run detection
		detector.run();

		if (opts & wbd::OPT_VERBOSE)
			std::cout << LIBHEADER << "Detection finished." << std::endl;
		
		if (opts & (wbd::OPT_VISUAL_OUTPUT|wbd::OPT_VISUAL_DEBUG))
		{
			cv::imshow(LIBNAME, image);			
            cv::waitKey(WB_WAIT_DELAY);
		}		
	}
	detector.free();
	video.release();

	if (opts & wbd::OPT_VERBOSE)
		std::cout << LIBHEADER << "Memory free." << std::endl;

	return true;
}

/**
 * @brief Chooses whether to process video or an image dataset.
 *
 * @param filename	filename
 * @param inputType	type of input
 * @param opts		run optseters
 */
bool process(std::string input, wbd::InputTypes inputType, wbd::RunSettings settings, uint32 opts = 0)
{	
    settings.inputType = inputType;
	switch (inputType)
	{
		case wbd::INPUT_IMAGE_DATASET:
		{
			processImageDataset(input, settings, opts);
			break;
		}
		case wbd::INPUT_VIDEO:
		{
			processVideo(input, settings, opts);
			break;
		}
		default:
			return false;
	}

	return true;
}

/**
 * @brief Basic program to test the detector.
 *
 * @param argc	number of arguments
 * @param argv	argument values
 * @return exit code
 *
 * @todo improve argument error handling
 */
int main(int argc, char** argv)
{
	std::string input;
	wbd::InputTypes mode;
	uint32 opts = 0;
	wbd::RunSettings settings;	

    // default detection mode

    int32 device_count, device = 0, multiproc_count = 0;
    cudaError res;
    res == cudaGetDeviceCount(&device_count);

    if (res != cudaErrorInsufficientDriver && res != cudaErrorNoDevice)
    {
        for (int32 i = 0; i < device_count; ++i)
        {
            cudaDeviceProp properties;
            GPU_CHECK_ERROR(cudaGetDeviceProperties(&properties, i));

            if (properties.multiProcessorCount > multiproc_count)
            {
                multiproc_count = properties.multiProcessorCount;
                device = i;
            }
        }

        cudaSetDevice(device);
        if (multiproc_count >= 2)
            settings.detectionMode = wbd::DET_ATOMIC_SHARED;
        else
            settings.detectionMode = wbd::DET_ATOMIC_GLOBAL;
    }
    else
        settings.detectionMode = wbd::DET_CPU;

    for (int32 i = 1; i < argc; ++i)
	{
		// input dataset
        if ((std::string(argv[i]) == "-D" || std::string(argv[i]) == "--dataset") && i + 1 < argc) {
			mode = wbd::INPUT_IMAGE_DATASET;
			input = argv[++i];
		}
		// input video
        else if ((std::string(argv[i]) == "-V" || std::string(argv[i]) == "--video") && i + 1 < argc) {
			mode = wbd::INPUT_VIDEO;
			input = argv[++i];
		}
		// output csv
        else if ((std::string(argv[i]) == "-c" || std::string(argv[i]) == "--csv") && i + 1 < argc) {
			settings.outputFilename = argv[++i];
			opts |= wbd::OPT_OUTPUT_CSV;
		}
		// verbose
        else if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--verbose")
			opts |= wbd::OPT_VERBOSE;
		// timer enabled
        else if (std::string(argv[i]) == "-t" || std::string(argv[i]) == "--timer")
			opts |= wbd::OPT_TIMER;
		// visual output
        else if (std::string(argv[i]) == "-o" || std::string(argv[i]) == "--visualoutput")
            opts |= wbd::OPT_VISUAL_OUTPUT;
        else if (std::string(argv[i]) == "-s" || std::string(argv[i]) == "--survivors")
            opts |= wbd::OPT_MEASURE_SURVIVORS;
		// visual debug
        else if (std::string(argv[i]) == "-d" || std::string(argv[i]) == "--visualdebug")
			opts |= (wbd::OPT_VISUAL_DEBUG|wbd::OPT_TIMER);
		// block size
        else if ((std::string(argv[i]) == "-b" || std::string(argv[i]) == "--blocksize") && i + 1 < argc)
			settings.blockSize = atoi(argv[++i]);
		// max frames processed
        else if ((std::string(argv[i]) == "-l" || std::string(argv[i]) == "--limitframes") && i + 1 < argc)
		{ 
			settings.maxFrames = atoi(argv[++i]);
			opts |= wbd::OPT_LIMIT_FRAMES;
		}
        // @todo remove this
        // DEPRACATRED
		// pyramid generation
		else if (std::string(argv[i]) == "-pg" && i + 1 < argc)
		{
			std::string str = argv[++i];
			if (str == "bindless")
				settings.pyGenMode = wbd::PYGEN_BINDLESS_TEXTURE;
			else 
			{
				std::cerr << LIBHEADER << "Option -pg (pyramid generation) has currently only one option available: 'bindless'." << std::endl;
				return EXIT_FAILURE;
			}
		}
        else if ((std::string(argv[i]) == "-m" || std::string(argv[i]) == "--detectionmode") && i + 1 < argc)
		{
			std::string str = argv[++i];
			if (str == "global")
				settings.detectionMode = wbd::DET_ATOMIC_GLOBAL;
			else if (str == "shared")
				settings.detectionMode = wbd::DET_ATOMIC_SHARED;
            else if (str == "prefixsum")
                settings.detectionMode = wbd::DET_PREFIXSUM;
            else if (str == "hybrid")
                settings.detectionMode = wbd::DET_HYBRIG_SG;
			else if (str == "cpu")
				settings.detectionMode = wbd::DET_CPU;
			else
			{
				std::cerr << LIBHEADER << "Option -d --detectionmode (detection mode) has only the following options available: 'global', 'shared', 'prefixsum', 'hybrid' and 'cpu'." << std::endl;
				return EXIT_FAILURE;
			}
		}
        // @todo remove this
        // DEPRECATED
		// pyramid type
        else if ((std::string(argv[i]) == "-p" || std::string(argv[i]) == "--pyramid") && i + 1 < argc)
		{
			std::string str = argv[++i];
			if (str == "horizontal")
				settings.pyType = wbd::PYTYPE_HORIZONAL;
			else if (str == "optimized")
				settings.pyType = wbd::PYTYPE_OPTIMIZED;
			else
			{
				std::cerr << LIBHEADER << "Option -p --pyramid (pyramid type) has only the following options available: 'horizontal' and 'optimized'." << std::endl;
				return EXIT_FAILURE;
			}
		}
		else {
            std::cerr << "An error occured during processing parameters." << std::endl;
			return EXIT_FAILURE;
		}
	}
	
	if (process(input, mode, settings, opts))
		return EXIT_SUCCESS;

	return EXIT_FAILURE;
}

