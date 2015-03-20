#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <fstream>
#include <string>
#include <iostream>
#include <chrono>

#include "wb_waldboostdetector.h"
#include "wb_enums.h"
#include "wb_general.h"
#include "wb_structures.h"

/**
 * @brief Processes an image dataset - a text file with a list of images.
 *
 * @param filename	filename
 * @param opts		run optseters
 */
bool processImageDataset(std::string filename, uint32 opts)
{	
	std::ifstream in;
	in.open(filename);

	std::string buffer;
	cv::Mat image;
	wb::WaldboostDetector detector;

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
		if (opts & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Initialized detector." << std::endl;

		detector.setImage(&image);
		if (opts & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Image set." << std::endl;

		if (opts & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Running detections ..." << std::endl;
		detector.run();
		if (opts & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Detection finished." << std::endl;

		if (opts & wb::OPT_VISUAL_OUTPUT)
		{
			cv::imshow(LIBNAME, image);
			cv::waitKey(WB_WAIT_DELAY);
		}

		detector.free();
		if (opts & wb::OPT_VERBOSE)
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
bool processVideo(std::string const& filename, wb::RunSettings const& settings, uint32 const& opts)
{
	cv::VideoCapture video;
	cv::Mat image;
	video.open(filename);

	// use first image just to init detector
	video >> image;
	if (image.empty())
		return false;

	wb::WaldboostDetector detector;	
	detector.setBlockSize(settings.blockSize, settings.blockSize);
	detector.setPyGenMode(settings.pyGenMode);
	detector.setPyType(settings.pyType);	
	detector.setRunOptions(opts);
	detector.setOutputFile(settings.outputFilename);
	detector.setDetectionMode(settings.detectionMode);
	detector.init(&image);

	if (opts & wb::OPT_VERBOSE)	
		std::cout << LIBHEADER << "Initialized detector." << std::endl;			

	while (true)
	{		
		if (opts & wb::OPT_LIMIT_FRAMES)
		{
			if (settings.maxFrames < detector.getFrameCount())
				break;
		}

		auto start_time = std::chrono::high_resolution_clock::now();
		video >> image;

		if (image.empty())
			break;

		// load detector with an image
		detector.setImage(&image);
		if (opts & wb::OPT_VERBOSE)
		{ 
			std::cout << LIBHEADER << "Image set." << std::endl;
			std::cout << LIBHEADER << "Running detections ..." << std::endl;
		}

		// run detection
		detector.run();

		if (opts & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Detection finished." << std::endl;
		
		if (opts & (wb::OPT_VISUAL_OUTPUT|wb::OPT_VISUAL_DEBUG))
		{
			cv::imshow(LIBNAME, image);
			cv::waitKey(WB_WAIT_DELAY);
		}		
	}
	detector.free();
	video.release();

	if (opts & wb::OPT_VERBOSE)
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
bool process(std::string input, wb::InputTypes inputType, wb::RunSettings settings, uint32 opts = 0)
{	
	switch (inputType)
	{
		case wb::INPUT_IMAGE_DATASET:
		{
			processImageDataset(input, opts);
			break;
		}
		case wb::INPUT_VIDEO:
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
	wb::InputTypes mode;
	uint32 opts = 0;
	wb::RunSettings settings;	
	for (int i = 1; i < argc; ++i)
	{
		// input dataset
		if (std::string(argv[i]) == "-id" && i + 1 < argc) {
			mode = wb::INPUT_IMAGE_DATASET;
			input = argv[++i];
		}
		// input video
		else if (std::string(argv[i]) == "-iv" && i + 1 < argc) {
			mode = wb::INPUT_VIDEO;
			input = argv[++i];
		}
		// output csv
		else if (std::string(argv[i]) == "-oc" && i + 1 < argc) {
			settings.outputFilename = argv[++i];
			opts |= wb::OPT_OUTPUT_CSV;
		}
		// verbose
		else if (std::string(argv[i]) == "-v")			
			opts |= wb::OPT_VERBOSE;
		// visual output
		else if (std::string(argv[i]) == "-t")		
			opts |= wb::OPT_TIMER;
		// visual output
		else if (std::string(argv[i]) == "-vo")
			opts |= wb::OPT_VISUAL_OUTPUT;
		// visual debug
		else if (std::string(argv[i]) == "-vd")
			opts |= (wb::OPT_VISUAL_DEBUG|wb::OPT_TIMER);
		// block size
		else if (std::string(argv[i]) == "-bs" && i + 1 < argc)
			settings.blockSize = atoi(argv[++i]);
		// max frames processed
		else if (std::string(argv[i]) == "-lf" && i + 1 < argc)
		{ 
			settings.maxFrames = atoi(argv[++i]);
			opts |= wb::OPT_LIMIT_FRAMES;
		}
		// pyramid generation
		else if (std::string(argv[i]) == "-pg" && i + 1 < argc)
		{
			std::string str = argv[++i];
			if (str == "single") 
				settings.pyGenMode = wb::PYGEN_SINGLE_TEXTURE;
			else if (str == "bindless")
				settings.pyGenMode = wb::PYGEN_BINDLESS_TEXTURE;
			else 
			{
				std::cerr << LIBHEADER << "Option -pg (pyramid generation) has two options available: 'bindless' and 'single'." << std::endl;
				return EXIT_FAILURE;
			}
		}
		else if (std::string(argv[i]) == "-dm" && i + 1 < argc)
		{
			std::string str = argv[++i];
			if (str == "aglobal")
				settings.detectionMode = wb::DET_ATOMIC_GLOBAL;
			else if (str == "ashared")
				settings.detectionMode = wb::DET_ATOMIC_SHARED;
			else if (str == "prefixsum")
				settings.detectionMode = wb::DET_PREFIXSUM;
			else
			{
				std::cerr << LIBHEADER << "Option -dm (detection mode) has only the following options available: 'aglobal', 'ashared' and 'prefixsum'." << std::endl;
				return EXIT_FAILURE;
			}
		}
		// pyramid type
		else if (std::string(argv[i]) == "-pt" && i + 1 < argc)
		{
			std::string str = argv[++i];
			if (str == "horizontal")
				settings.pyType = wb::PYTYPE_HORIZONAL;
			else if (str == "optimized")
				settings.pyType = wb::PYTYPE_OPTIMIZED;
			else
			{
				std::cerr << LIBHEADER << "Option -pt (pyramid type) has two options available: 'horizontal' and 'optimized'." << std::endl;
				return EXIT_FAILURE;
			}
		}
		else {
			std::cerr << "Usage: " << argv[0] << "-id [input dataset] or -iv [input video]" << std::endl;
			return EXIT_FAILURE;
		}
	}
	
	if (process(input, mode, settings, opts))
		return EXIT_SUCCESS;

	return EXIT_FAILURE;
}

