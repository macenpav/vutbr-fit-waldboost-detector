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
			std::cerr << LIBHEADER << "Could not open or find the image (filename: " << filename.c_str() << ")" << std::endl;
			continue;
		}

		detector.init(&image);
		if (param & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Initialized detector." << std::endl;

		detector.setImage(&image);
		if (param & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Image set." << std::endl;

		if (param & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Running detections ..." << std::endl;
		detector.run();
		if (param & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Detection finished." << std::endl;

		if (param & wb::OPT_VISUAL_OUTPUT)
		{
			cv::imshow(LIBNAME, image);
			cv::waitKey(WB_WAIT_DELAY);
		}

		detector.free();
		if (param & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Memory free." << std::endl;		
	}

	return true;
}

/**
* @brief Processes a video.
*
* @param filename	filename
* @param param		run parameters
* @todo implement pyType
*/
bool processVideo(std::string const& filename, uint32 const& param, std::string const& output, wb::PyramidGenModes const& pyGenMode)
{
	cv::VideoCapture video;
	cv::Mat image;
	video.open(filename);

	// use first image just to init detector
	video >> image;
	if (image.empty())
		return false;

	wb::WaldboostDetector detector;	
	detector.setBlockSize(32, 32);
	detector.setPyGenMode(pyGenMode);
	detector.setPyType(wb::PYTYPE_OPTIMIZED);
	detector.setRunParameters(param);
	detector.setOutputFile(output);
	detector.init(&image);

	if (param & wb::OPT_VERBOSE)	
		std::cout << LIBHEADER << "Initialized detector." << std::endl;			

	while (true)
	{		
		auto start_time = std::chrono::high_resolution_clock::now();
		video >> image;

		if (image.empty())
			break;

		// load detector with an image
		detector.setImage(&image);
		if (param & wb::OPT_VERBOSE)
		{ 
			std::cout << LIBHEADER << "Image set." << std::endl;
			std::cout << LIBHEADER << "Running detections ..." << std::endl;
		}

		// run detection
		detector.run();

		if (param & wb::OPT_VERBOSE)
			std::cout << LIBHEADER << "Detection finished." << std::endl;
		
		if (param & wb::OPT_VISUAL_OUTPUT)
		{
			cv::imshow(LIBNAME, image);
			cv::waitKey(WB_WAIT_DELAY);
		}		
	}
	detector.free();
	video.release();

	if (param & wb::OPT_VERBOSE)
		std::cout << LIBHEADER << "Memory free." << std::endl;

	return true;
}

/**
* @brief Chooses whether to process video or an image dataset.
*
* @param filename	filename
* @param inputType	type of input
* @param param		run parameters
*/
bool process(std::string input, wb::InputTypes inputType, uint32 param = 0, std::string output = std::string(""), wb::PyramidGenModes pyGenMode = wb::PYGEN_BINDLESS_TEXTURE)
{	
	switch (inputType)
	{
		case wb::INPUT_IMAGE_DATASET:
		{
			processImageDataset(input, param);
			break;
		}
		case wb::INPUT_VIDEO:
		{
			processVideo(input, param, output, pyGenMode);
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
	std::string input, output;
	wb::InputTypes mode;
	uint32 param = 0;
	wb::PyramidGenModes pyGenMode = wb::MAX_PYGEN_MODES;
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
			output = argv[++i];
			param |= wb::OPT_OUTPUT_CSV;
		}
		// verbose
		else if (std::string(argv[i]) == "-v") {			
			param |= wb::OPT_VERBOSE;
		}
		// visual output
		else if (std::string(argv[i]) == "-t") {			
			param |= wb::OPT_TIMER;
		}
		// visual output
		else if (std::string(argv[i]) == "-vo") {
			param |= wb::OPT_VISUAL_OUTPUT;
		}
		// visual debug
		else if (std::string(argv[i]) == "-vd") {
			param |= wb::OPT_VISUAL_DEBUG | wb::OPT_TIMER;
		}
		else if (std::string(argv[i]) == "-pg" && i + 1 < argc)
		{
			std::string str = argv[++i];
			if (str == "single") {
				pyGenMode = wb::PYGEN_SINGLE_TEXTURE;
			}
			else if (str == "bindless")
			{
				pyGenMode = wb::PYGEN_BINDLESS_TEXTURE;
			}
			else 
			{
				std::cerr << LIBHEADER << "Option -pg (pyramid generation) has two options available: 'bindless' and 'single'." << std::endl;
				return EXIT_FAILURE;
			}
		}
		else {
			std::cerr << "Usage: " << argv[0] << "-id [input dataset] or -iv [input video]" << std::endl;
			return EXIT_FAILURE;
		}
	}
	
	if (process(input, mode, param, output, pyGenMode))
		return EXIT_SUCCESS;

	return EXIT_FAILURE;
}

