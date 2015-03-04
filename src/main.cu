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

const std::string LIBNAME = "waldboost-detector";

/**
* @brief Processes an image dataset - a text file with a list of images.
*
* @param filename	filename
* @param param		run parameters
*/
bool processImageDataset(std::string filename, uint32 param)
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
			std::cerr << "[" << LIBNAME << "]: " << "Could not open or find the image (filename: " << filename.c_str() << ")" << std::endl;
			continue;
		}

		detector.init(&image);
		if (param & wb::OPT_VERBOSE)
			std::cout << "[" << LIBNAME << "]: " << "Initialized detector." << std::endl;

		detector.setImage(&image);
		if (param & wb::OPT_VERBOSE)
			std::cout << "[" << LIBNAME << "]: " << "Image set." << std::endl;

		if (param & wb::OPT_VERBOSE)
			std::cout << "[" << LIBNAME << "]: " << "Running detections ..." << std::endl;
		detector.run();
		if (param & wb::OPT_VERBOSE)
			std::cout << "[" << LIBNAME << "]: " << "Detection finished." << std::endl;

		if (param & wb::OPT_VISUAL_OUTPUT)
		{
			cv::imshow(LIBNAME, image);
			cv::waitKey(WAIT_DELAY);
		}

		detector.free();
		if (param & wb::OPT_VERBOSE)
			std::cout << "[" << LIBNAME << "]: " << "Memory free." << std::endl;		
	}

	return true;
}

/**
* @brief Processes a video.
*
* @param filename	filename
* @param param		run parameters
*/
bool processVideo(std::string filename, uint32 param)
{
	cv::VideoCapture video;
	cv::Mat image;
	video.open(filename);

	// use first image just to init detector
	video >> image;
	if (image.empty())
		return false;

	wb::WaldboostDetector detector;
	detector.init(&image);

	if (param & wb::OPT_VERBOSE)	
		std::cout << "[" << LIBNAME << "]: " << "Initialized detector." << std::endl;			

	while (true)
	{		
		auto start_time = std::chrono::high_resolution_clock::now();
		video >> image;

		if (image.empty())
			break;

		// load detector with an image
		detector.setImage(&image);
		if (param & wb::OPT_VERBOSE)
			std::cout << "[" << LIBNAME << "]: " << "Image set." << std::endl;

		if (param & wb::OPT_VERBOSE)
			std::cout << "[" << LIBNAME << "]: " << "Running detections ..." << std::endl;

		// run detection
		detector.run();

		if (param & wb::OPT_VERBOSE)
			std::cout << "[" << LIBNAME << "]: " << "Detection finished." << std::endl;
		
		if (param & wb::OPT_VISUAL_OUTPUT)
		{
			cv::imshow(LIBNAME, image);
			cv::waitKey(WAIT_DELAY);
		}
		auto end_time = std::chrono::high_resolution_clock::now();
		std::cout << "TOTAL TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " FPS: " << 1000 / std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
	}
	detector.free();
	video.release();

	if (param & wb::OPT_VERBOSE)
		std::cout << "[" << LIBNAME << "]: " << "Memory free." << std::endl;

	return true;
}

/**
* @brief Chooses whether to process video or an image dataset.
*
* @param filename	filename
* @param inputType	type of input
* @param param		run parameters
*/
bool process(std::string filename, wb::WBInputTypes inputType, uint32 param)
{	
	switch (inputType)
	{
		case wb::INPUT_IMAGE_DATASET:
		{
			processImageDataset(filename, param);
			break;
		}
		case wb::INPUT_VIDEO:
		{
			processVideo(filename, param);
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
*/
int main(int argc, char** argv)
{
	std::string filename;
	wb::WBInputTypes mode;
	for (int i = 1; i < argc; ++i)
	{
		if (std::string(argv[i]) == "-id" && i + 1 < argc) {
			mode = wb::INPUT_IMAGE_DATASET;
			filename = argv[++i];
		}
		else if (std::string(argv[i]) == "-iv" && i + 1 < argc) {
			mode = wb::INPUT_VIDEO;
			filename = argv[++i];
		}
		else {
			std::cerr << "Usage: " << argv[0] << "-id [input dataset] or -iv [input video]" << std::endl;
			return EXIT_FAILURE;
		}
	}

	uint32 param = wb::OPT_ALL;
	if (process(filename, mode, param))
		return EXIT_SUCCESS;

	return EXIT_FAILURE;
}

