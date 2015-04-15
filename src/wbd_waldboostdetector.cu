#include "wbd_waldboostdetector.cuh"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include "cuda_runtime.h"

#include "wbd_general.h"
#include "wbd_structures.h"
#include "wbd_detector.h"
#include "wbd_alphas.h"
#include "wbd_simple.h"
#include "wbd_func.h"
#include "wbd_gpu.cuh"
#include "wbd_gpu_pyramid.cuh"
#include "wbd_gpu_detection.cuh"

namespace wbd 
{			
	WaldboostDetector::WaldboostDetector()
	{
		gpu::detection::initDetectionStages();
	}

	void WaldboostDetector::_pyramidGenBindlessTexture()
	{
		Octave octave0 = _pyramid.octaves[0];

		dim3 grid0(octave0.width / _kernelBlockConfig[KERTYPE_PYRAMID].x + 1, octave0.height / _kernelBlockConfig[KERTYPE_PYRAMID].y + 1, 1);
		gpu::pyramid::createFirstPyramid<<<grid0, _kernelBlockConfig[KERTYPE_PYRAMID]>>>(_devPyramidImage[0], _devPyramidData, _preprocessedImageTexture, octave0.width, octave0.height, _info.width, _info.height, _pyramid.canvasWidth, WB_LEVELS_PER_OCTAVE);
		GPU_CHECK_ERROR(cudaPeekAtLastError());
		
		bindFloatImageTo2DTexture(&_texturePyramidObjects[0], _devPyramidImage[0], octave0.width, octave0.height);		

		for (uint8 oct = 1; oct < WB_OCTAVES; ++oct)
		{
			Octave octave = _pyramid.octaves[oct];

			dim3 grid(octave.width / _kernelBlockConfig[KERTYPE_PYRAMID].x + 1, octave.height / _kernelBlockConfig[KERTYPE_PYRAMID].y + 1, 1);
			gpu::pyramid::createPyramidFromPyramid<<<grid, _kernelBlockConfig[KERTYPE_PYRAMID]>>>(_devPyramidImage[oct], _devPyramidData, _texturePyramidObjects[oct - 1], octave.width, octave.height, octave.offsetX, octave.offsetY, _pyramid.canvasWidth);
			GPU_CHECK_ERROR(cudaPeekAtLastError());

			if (oct != WB_OCTAVES - 1)			
				bindFloatImageTo2DTexture(&_texturePyramidObjects[oct], _devPyramidImage[oct], _pyramid.octaves[oct].width, _pyramid.octaves[oct].height);
		}
	}

	void WaldboostDetector::_pyramidKernelWrapper()
	{
		switch (_pyGenMode)
		{			
			case PYGEN_BINDLESS_TEXTURE:
				_pyramidGenBindlessTexture();
				break;

			default:
				break;
		}					
	}

	void WaldboostDetector::_precalcHorizontalPyramid()
	{		
		uint32 currentOffsetX = 0;		
		
		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
		{		
			Octave octave;

			octave.width = _info.width >> oct;			
			_pyramid.canvasWidth += octave.width;

			float scale = powf(2.f, static_cast<float>(oct));
			uint32 currentOffsetY = 0;			
			for (uint8 lvl = 0; lvl < WB_LEVELS_PER_OCTAVE; ++lvl)
			{
				PyramidImage data;

				float scaledWidth = _info.width / scale;
				float scaledHeight = _info.height / scale;

				data.width = static_cast<uint32>(scaledWidth);
				data.height = static_cast<uint32>(scaledHeight);
				data.offsetY = currentOffsetY;
				data.offsetX = currentOffsetX;

				scale *= WB_SCALING_FACTOR;
				currentOffsetY += data.height;				

				octave.images[lvl] = data;
			}

			if (oct == 0) {
				_pyramid.canvasHeight = currentOffsetY;				
			}			

			octave.height = currentOffsetY;
			octave.imageSize = octave.height * octave.width;			
			currentOffsetX += octave.width;

			_pyramid.octaves[oct] = octave;
		}
		
		_pyramid.canvasImageSize = _pyramid.canvasWidth * _pyramid.canvasHeight;
	}

	void WaldboostDetector::_precalc4x8Pyramid()
	{
		_pyramid.canvasWidth = _info.width + (_info.width >> 1);

		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
		{
			Octave octave;

			octave.width = _info.width >> oct;

			float scale = powf(2.f, static_cast<float>(oct));
			switch (oct)
			{
				case 0:
					octave.offsetX = 0;
					octave.offsetY = 0;
					break;
				case 1:
					octave.offsetX = _pyramid.octaves[0].width;
					octave.offsetY = 0;
					break;
				case 2:
					octave.offsetX = _pyramid.octaves[1].offsetX;
					octave.offsetY = _pyramid.octaves[1].height;
					break;
				case 3:
					octave.offsetX = _pyramid.octaves[2].offsetX + _pyramid.octaves[2].width;
					octave.offsetY = _pyramid.octaves[2].offsetY;
					break;
				default:
					break;
			}

			uint32 currentOffsetY = octave.offsetY;
			uint32 currentHeight = 0;
			for (uint8 lvl = 0; lvl < WB_LEVELS_PER_OCTAVE; ++lvl)
			{
				PyramidImage data;

				float scaledWidth = _info.width / scale;
				float scaledHeight = _info.height / scale;

				data.width = static_cast<uint32>(scaledWidth);
				data.height = static_cast<uint32>(scaledHeight);
				data.offsetY = currentOffsetY;
				data.offsetX = octave.offsetX;

				scale *= WB_SCALING_FACTOR;
				currentOffsetY += data.height;
				currentHeight += data.height;

				octave.images[lvl] = data;
			}

			octave.height = currentHeight;
			octave.imageSize = octave.width * octave.height;

			_pyramid.octaves[oct] = octave;
		}

		_pyramid.canvasHeight = _pyramid.octaves[0].height;
		_pyramid.canvasImageSize = _pyramid.canvasWidth * _pyramid.canvasHeight;		
	}	

	void WaldboostDetector::init(cv::Mat* image)
	{
		_info.width = image->cols;
		_info.height = image->rows;
		_info.imageSize = image->cols * image->rows;		
		_info.channels = image->channels();	

		_frame = 0;

		ClockPoint init_time_start, init_time_end;
		if (_opt & OPT_TIMER)
		{ 
			_initTimers();
			init_time_start = Clock::now();
		}
	
		switch (_pyType)
		{
			case PYTYPE_OPTIMIZED:
				_precalc4x8Pyramid();
				break;

			case PYTYPE_HORIZONAL:
				_precalcHorizontalPyramid();
				break;
		}

		if (_opt & OPT_TIMER)
		{
			init_time_end = Clock::now();
			FPDuration duration = init_time_end - init_time_start;
			_timers[TIMER_INIT] += static_cast<float>(std::chrono::duration_cast<Nanoseconds>(duration).count()) / 1000000.f;
		}
		
		if (_opt & OPT_VERBOSE)
		{
			std::cout << LIBHEADER << "Finished generating pyramid." << std::endl;
			std::cout << LIBHEADER << "Outputting details ..." << std::endl;
			std::cout << LIBHEADER << "Canvas size: " << _pyramid.canvasImageSize << " (" << _pyramid.canvasWidth << "x" << _pyramid.canvasHeight << ")" << std::endl;
			std::cout << LIBHEADER << "Octaves ..." << std::endl;

			for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
			{
				Octave octave = _pyramid.octaves[oct];
				std::cout << LIBHEADER << "[" << (uint32)oct << "]" << std::endl;
				std::cout << LIBHEADER << "Octave size: " << octave.imageSize << " (" << octave.width << "x" << octave.height << ")" << std::endl;
				std::cout << LIBHEADER << "Octave offsets x: " << octave.offsetX << " y: " << octave.offsetY << std::endl;
				std::cout << LIBHEADER << "Images ..." << std::endl;

				for (uint8 lvl = 0; lvl < WB_LEVELS_PER_OCTAVE; ++lvl)
				{
					PyramidImage im = _pyramid.octaves[oct].images[lvl];
					std::cout << LIBHEADER << "[" << (uint32)oct << "]" << "[" << (uint32)lvl << "]" << std::endl;
					std::cout << LIBHEADER << "Image size: " << im.width << "x" << im.height << std::endl;
					std::cout << LIBHEADER << "Image offsets x: " << im.offsetX << " y: " << im.offsetY << std::endl;
				}
				std::cout << std::endl;
			}
		}

		if (_opt & OPT_TIMER)
		{
			init_time_start = Clock::now();
		}		

		GPU_CHECK_ERROR(cudaMalloc((void**)&_devOriginalImage, sizeof(uint8) * _info.imageSize * _info.channels));
		GPU_CHECK_ERROR(cudaMalloc((void**)&_devPreprocessedImage, sizeof(float) * _info.imageSize));
				
		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
			GPU_CHECK_ERROR(cudaMalloc((void**)&_devPyramidImage[oct], sizeof(float) * _pyramid.octaves[oct].imageSize));

		GPU_CHECK_ERROR(cudaMalloc((void**)&_devPyramidData, sizeof(float) * _pyramid.canvasImageSize));

		GPU_CHECK_ERROR(cudaMalloc((void**)&_devDetections, sizeof(Detection) * WB_MAX_DETECTIONS));
		GPU_CHECK_ERROR(cudaMalloc((void**)&_devDetectionCount, sizeof(uint32)));
		GPU_CHECK_ERROR(cudaMemset(_devDetectionCount, 0x00, sizeof(uint32)));
		GPU_CHECK_ERROR(cudaMalloc((void**)&_devSurvivors, sizeof(SurvivorData) * (_pyramid.canvasWidth / _kernelBlockConfig[KERTYPE_DETECTION].x + 1) * (_pyramid.canvasHeight / _kernelBlockConfig[KERTYPE_DETECTION].y + 1) * (_kernelBlockConfig[KERTYPE_DETECTION].x * _kernelBlockConfig[KERTYPE_DETECTION].y)));
		GPU_CHECK_ERROR(cudaMalloc((void**)&_devSurvivorCount, sizeof(uint32)));
		GPU_CHECK_ERROR(cudaMemset(_devSurvivorCount, 0x00, sizeof(uint32)));
				
		GPU_CHECK_ERROR(cudaMalloc(&_devAlphaBuffer, WB_STAGE_COUNT * WB_ALPHA_COUNT * sizeof(float)));
		GPU_CHECK_ERROR(cudaMemcpy(_devAlphaBuffer, alphas, WB_STAGE_COUNT * WB_ALPHA_COUNT * sizeof(float), cudaMemcpyHostToDevice));
		
		bindLinearFloatDataToTexture(&_alphasTexture, _devAlphaBuffer, WB_STAGE_COUNT * WB_ALPHA_COUNT);
		bindFloatImageTo2DTexture(&_finalPyramidTexture, _devPyramidData, _pyramid.canvasWidth, _pyramid.canvasHeight);

		GPU_CHECK_ERROR(cudaDeviceSynchronize());

		if (_opt & OPT_TIMER)
		{
			init_time_end = Clock::now();
			FPDuration duration = init_time_end - init_time_start;
			_timers[TIMER_INIT] += static_cast<float>(std::chrono::duration_cast<Nanoseconds>(duration).count()) / 1000000.f;
		}

		if (_opt & OPT_OUTPUT_CSV)
		{
			std::ofstream file;
			file.open(_outputFilename, std::ios::out);
			file << "init;" << _timers[TIMER_INIT] << std::endl << std::endl;
			file << "frame;preprocessing;pyramid gen.;detection" << std::endl;
			file.close();
		}
	}	

	void WaldboostDetector::setImage(cv::Mat* image)
	{
		_initTimers();		

		_myImage = image;
		cudaMemcpy(_devOriginalImage, image->data, _info.imageSize * _info.channels * sizeof(uint8), cudaMemcpyHostToDevice);

		dim3 grid(_info.width / _kernelBlockConfig[KERTYPE_PREPROCESS].x + 1, _info.height / _kernelBlockConfig[KERTYPE_PREPROCESS].y + 1, 1);

		cudaEvent_t start_preprocess, stop_preprocess;
		if (_opt & OPT_TIMER)
		{
			GPU_CHECK_ERROR(cudaEventCreate(&start_preprocess));
			GPU_CHECK_ERROR(cudaEventCreate(&stop_preprocess));
			GPU_CHECK_ERROR(cudaEventRecord(start_preprocess));
		}

		gpu::preprocess<<<grid, _kernelBlockConfig[KERTYPE_PREPROCESS]>>>(_devPreprocessedImage, _devOriginalImage, _info.width, _info.height);
		GPU_CHECK_ERROR(cudaPeekAtLastError());

		if (_opt & OPT_TIMER)
		{
			GPU_CHECK_ERROR(cudaEventRecord(stop_preprocess));
			GPU_CHECK_ERROR(cudaEventSynchronize(stop_preprocess));
			GPU_CHECK_ERROR(cudaEventElapsedTime(&_timers[TIMER_PREPROCESS], start_preprocess, stop_preprocess));
		}

		if (_opt & OPT_VISUAL_DEBUG)
		{
			cv::Mat tmp(cv::Size(_info.width, _info.height), CV_32FC1);
			GPU_CHECK_ERROR(cudaMemcpy(tmp.data, _devPreprocessedImage, _info.imageSize * sizeof(float), cudaMemcpyDeviceToHost));
			cv::imshow("Preprocessed image (B&W image should be displayed)", tmp);
			cv::waitKey(WB_WAIT_DELAY);
		}

		bindFloatImageTo2DTexture(&_preprocessedImageTexture, _devPreprocessedImage, _info.width, _info.height);

		cudaEvent_t start_pyramid, stop_pyramid;			
		if (_opt & OPT_TIMER)
		{
			GPU_CHECK_ERROR(cudaEventCreate(&start_pyramid));
			GPU_CHECK_ERROR(cudaEventCreate(&stop_pyramid));
			GPU_CHECK_ERROR(cudaEventRecord(start_pyramid));
		}

		_pyramidKernelWrapper();		
		
		if (_opt & OPT_TIMER)
		{			
			GPU_CHECK_ERROR(cudaEventRecord(stop_pyramid));
			GPU_CHECK_ERROR(cudaEventSynchronize(stop_pyramid));
			GPU_CHECK_ERROR(cudaEventElapsedTime(&_timers[TIMER_PYRAMID], start_pyramid, stop_pyramid));
		}
		
		if (_opt & OPT_VISUAL_DEBUG)
		{	
			// copy image from GPU to CPU
			float* pyramidImage;			
			dim3 grid(_pyramid.canvasWidth / _kernelBlockConfig[KERTYPE_PYRAMID].x + 1, _pyramid.canvasHeight / _kernelBlockConfig[KERTYPE_PYRAMID].y + 1, 1);
			GPU_CHECK_ERROR(cudaMalloc((void**)&pyramidImage, _pyramid.canvasImageSize * sizeof(float)));
			
			// copies from statically defined texture
			gpu::copyImageFromTextureObject<<<grid, _kernelBlockConfig[KERTYPE_PYRAMID]>>>(pyramidImage, _finalPyramidTexture, _pyramid.canvasWidth, _pyramid.canvasHeight);
			GPU_CHECK_ERROR(cudaPeekAtLastError());

			// display using OpenCV
			cv::Mat tmp(cv::Size(_pyramid.canvasWidth, _pyramid.canvasHeight), CV_32FC1);
			GPU_CHECK_ERROR(cudaMemcpy(tmp.data, pyramidImage, _pyramid.canvasImageSize * sizeof(float), cudaMemcpyDeviceToHost));
			cv::imshow("Pyramid texture (B&W pyramid images should be displayed)", tmp);
			cv::waitKey(WB_WAIT_DELAY);			

			GPU_CHECK_ERROR(cudaFree(pyramidImage));
		}
	}

	void WaldboostDetector::_initTimers()
	{
		for (uint8 i = 0; i < MAX_TIMERS; ++i)
			_timers[i] = 0.f;
	}

	void WaldboostDetector::_processDetections()
	{
		Detection detections[WB_MAX_DETECTIONS];
		uint32 detectionCount = 0;
		GPU_CHECK_ERROR(cudaMemcpy(&detectionCount, _devDetectionCount, sizeof(uint32), cudaMemcpyDeviceToHost));
		GPU_CHECK_ERROR(cudaMemcpy(&detections, _devDetections, detectionCount * sizeof(Detection), cudaMemcpyDeviceToHost));

		_processDetections(detections, detectionCount);
	}

	void WaldboostDetector::_processDetections(Detection* detections, uint32 const& detectionCount)
	{		
		if (_opt & OPT_VERBOSE)	
			std::cout << LIBHEADER << "Detection count: " << detectionCount << std::endl;		

		if (_opt & (OPT_VISUAL_DEBUG|OPT_VISUAL_OUTPUT))
		{ 
			std::string t = std::string("Detection count: ") + std::to_string(detectionCount);
			cv::putText(*_myImage, t, cv::Point(10, 65), cv::FONT_HERSHEY_SIMPLEX, 0.35, CV_RGB(0, 255, 0));
		}

		for (uint32 i = 0; i < detectionCount; ++i)
		{
			Detection d = detections[i];

			uint8 oct, lvl;
			for (oct = 0; oct < WB_OCTAVES; ++oct)
			{
				Octave octave = _pyramid.octaves[oct];
				if (d.x >= octave.offsetX && d.x < octave.offsetX + octave.width && 
					d.y >= octave.offsetY && d.y < octave.offsetY + octave.height)
				{
					for (lvl = 0; lvl < WB_LEVELS_PER_OCTAVE; ++lvl)
					{
						PyramidImage pi = octave.images[lvl];
						if (d.x >= pi.offsetX && d.x < pi.offsetX + pi.width &&
							d.y >= pi.offsetY && d.y < pi.offsetY + pi.height)
							break;
					}
					break;
				}
			}

			float scale = pow(2.f, static_cast<float>(oct)+1.f / static_cast<float>(WB_LEVELS_PER_OCTAVE)* static_cast<float>(lvl));

			d.width = static_cast<uint32>(static_cast<float>(d.width) * scale);
			d.height = static_cast<uint32>(static_cast<float>(d.height) * scale);

			d.x -= (_pyramid.octaves[oct].images[lvl].offsetX);
			d.x = static_cast<uint32>(static_cast<float>(d.x) * scale);

			d.y -= (_pyramid.octaves[oct].images[lvl].offsetY);
			d.y = static_cast<uint32>(static_cast<float>(d.y) * scale);

			if (_opt & (OPT_VISUAL_DEBUG|OPT_VISUAL_OUTPUT))
				cv::rectangle(*_myImage, cvPoint(d.x, d.y), cvPoint(d.x + d.width, d.y + d.height), CV_RGB(0, 255, 0));
		}	
	}

	void WaldboostDetector::run()
	{		
		if (_opt & OPT_VERBOSE)		
			std::cout << LIBHEADER << "Processing detections ..." << std::endl;		
		
		dim3 grid(_pyramid.canvasWidth / _kernelBlockConfig[KERTYPE_DETECTION].x + 1, _pyramid.canvasHeight / _kernelBlockConfig[KERTYPE_DETECTION].y + 1, 1);
		GPU_CHECK_ERROR(cudaMemset(_devDetectionCount, 0, sizeof(uint32)));
			
		cudaEvent_t start_detection, stop_detection;		
		switch (_detectionMode)
		{
			case DET_ATOMIC_GLOBAL:
				if (_opt & OPT_TIMER)
				{
					GPU_CHECK_ERROR(cudaEventCreate(&start_detection));
					GPU_CHECK_ERROR(cudaEventCreate(&stop_detection));
					GPU_CHECK_ERROR(cudaEventRecord(start_detection));
				}
				gpu::detection::atomicglobal::detect<<<grid, _kernelBlockConfig[KERTYPE_DETECTION]>>>(_finalPyramidTexture, _alphasTexture, _pyramid.canvasWidth, _pyramid.canvasHeight, _devSurvivors, _devDetections, _devDetectionCount);
				GPU_CHECK_ERROR(cudaPeekAtLastError());
				if (_opt & OPT_TIMER)
				{
					GPU_CHECK_ERROR(cudaEventRecord(stop_detection));
					GPU_CHECK_ERROR(cudaEventSynchronize(stop_detection));
					GPU_CHECK_ERROR(cudaEventElapsedTime(&_timers[TIMER_DETECTION], start_detection, stop_detection));
				}
				_processDetections();
				break;

			case DET_ATOMIC_SHARED:
			{
				if (_opt & OPT_TIMER)
				{
					GPU_CHECK_ERROR(cudaEventCreate(&start_detection));
					GPU_CHECK_ERROR(cudaEventCreate(&stop_detection));
					GPU_CHECK_ERROR(cudaEventRecord(start_detection));
				}
				uint32 sharedMemsize = _kernelBlockConfig[KERTYPE_DETECTION].x * _kernelBlockConfig[KERTYPE_DETECTION].y * sizeof(SurvivorData);
				gpu::detection::atomicshared::detect <<<grid, _kernelBlockConfig[KERTYPE_DETECTION], sharedMemsize>>>(_finalPyramidTexture, _alphasTexture, _pyramid.canvasWidth, _pyramid.canvasHeight, _devDetections, _devDetectionCount);
				GPU_CHECK_ERROR(cudaPeekAtLastError());
				if (_opt & OPT_TIMER)
				{
					GPU_CHECK_ERROR(cudaEventRecord(stop_detection));
					GPU_CHECK_ERROR(cudaEventSynchronize(stop_detection));
					GPU_CHECK_ERROR(cudaEventElapsedTime(&_timers[TIMER_DETECTION], start_detection, stop_detection));
				}
				_processDetections();
				break;
			}
			case DET_PREFIXSUM:
			{
				if (_opt & OPT_TIMER)
				{
					GPU_CHECK_ERROR(cudaEventCreate(&start_detection));
					GPU_CHECK_ERROR(cudaEventCreate(&stop_detection));
					GPU_CHECK_ERROR(cudaEventRecord(start_detection));
				}
				uint32 sharedMemsize = (_kernelBlockConfig[KERTYPE_DETECTION].x * _kernelBlockConfig[KERTYPE_DETECTION].y * sizeof(SurvivorData)) + (_kernelBlockConfig[KERTYPE_DETECTION].x * _kernelBlockConfig[KERTYPE_DETECTION].y * sizeof(uint32));
				gpu::detection::prefixsum::detect <<<grid, _kernelBlockConfig[KERTYPE_DETECTION], sharedMemsize>>>(_finalPyramidTexture, _alphasTexture, _devDetections, _devDetectionCount);
				GPU_CHECK_ERROR(cudaPeekAtLastError());
				if (_opt & OPT_TIMER)
				{
					GPU_CHECK_ERROR(cudaEventRecord(stop_detection));
					GPU_CHECK_ERROR(cudaEventSynchronize(stop_detection));
					GPU_CHECK_ERROR(cudaEventElapsedTime(&_timers[TIMER_DETECTION], start_detection, stop_detection));
				}
				_processDetections();
				break;
			}
			case DET_CPU:
			{
				// copy image from texture to GPU mem and then to CPU mem
				float* pyramidImage;
				dim3 grid(_pyramid.canvasWidth / _kernelBlockConfig[KERTYPE_PYRAMID].x + 1, _pyramid.canvasHeight / _kernelBlockConfig[KERTYPE_PYRAMID].y + 1, 1);
				GPU_CHECK_ERROR(cudaMalloc((void**)&pyramidImage, _pyramid.canvasImageSize * sizeof(float)));
				gpu::copyImageFromTextureObject<<<grid, _kernelBlockConfig[KERTYPE_PYRAMID]>>>(pyramidImage, _finalPyramidTexture, _pyramid.canvasWidth, _pyramid.canvasHeight);

				// float representation is used inside GPU
				// we need to convert it to uint8
				cv::Mat tmp(cv::Size(_pyramid.canvasWidth, _pyramid.canvasHeight), CV_32FC1);	
				uint8* bw = (uint8*)malloc(_pyramid.canvasWidth * _pyramid.canvasHeight);
				
				GPU_CHECK_ERROR(cudaMemcpy(tmp.data, pyramidImage, _pyramid.canvasImageSize * sizeof(float), cudaMemcpyDeviceToHost));
				for (int i = 0; i < tmp.rows; i++)
					for (int j = 0; j < tmp.cols; j++)						
						bw[i * tmp.cols + j] = static_cast<uint8>(tmp.at<float>(i, j) * 255.f);				
				GPU_CHECK_ERROR(cudaFree(pyramidImage));

				ClockPoint det_time_start, det_time_end;
				if (_opt & OPT_TIMER)
					det_time_start = Clock::now();

				std::vector<Detection> detections;
				simple::detect(detections, bw, _pyramid.canvasWidth, _pyramid.canvasHeight, _pyramid.canvasWidth);
				::free(bw);

				if (_opt & OPT_TIMER)
				{
					det_time_end = Clock::now();
					FPDuration duration = det_time_end - det_time_start;
					_timers[TIMER_DETECTION] += static_cast<float>(std::chrono::duration_cast<Nanoseconds>(duration).count()) / 1000000.f;
				}
				_processDetections(&(detections[0]), detections.size());
			}
			break;
		}														

		if (_opt & OPT_TIMER)
		{			
			if (_opt & OPT_VERBOSE)
			{ 
				std::cout << "Preprocessing: " << _timers[TIMER_PREPROCESS] << std::endl;
				std::cout << "Pyramid gen.: " << _timers[TIMER_PYRAMID] << std::endl;
				std::cout << "Detection: " << _timers[TIMER_DETECTION] << std::endl;
			}

			if (_opt & (OPT_VISUAL_OUTPUT | OPT_VISUAL_DEBUG))
			{
				std::string t1 = std::string("Preprocessing: ") + std::to_string(_timers[TIMER_PREPROCESS]) + std::string(" ms");
				std::string t2 = std::string("Pyramid gen.: ") + std::to_string(_timers[TIMER_PYRAMID]) + std::string(" ms");
				std::string t3 = std::string("Detection: ") + std::to_string(_timers[TIMER_DETECTION]) + std::string(" ms");
				cv::putText(*_myImage, t1, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.35, CV_RGB(0, 255, 0));
				cv::putText(*_myImage, t2, cv::Point(10, 35), cv::FONT_HERSHEY_SIMPLEX, 0.35, CV_RGB(0, 255, 0));
				cv::putText(*_myImage, t3, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.35, CV_RGB(0, 255, 0));
			}

			if (_opt & OPT_OUTPUT_CSV)
			{				
				std::ofstream file;
				file.open(_outputFilename, std::ios::out|std::ios::app);
				file << _frame << ";" << _timers[TIMER_PREPROCESS] << ";" << _timers[TIMER_PYRAMID] << ";" << _timers[TIMER_DETECTION] << std::endl;
				file.close();				
			}
		}		
		_frame++;		
	}

	void WaldboostDetector::free()
	{
		for (uint8 oct = 0; oct < WB_OCTAVES; ++oct)
		{ 
			GPU_CHECK_ERROR(cudaFree(_devPyramidImage[oct]));
			GPU_CHECK_ERROR(cudaDestroyTextureObject(_texturePyramidObjects[oct]));
		}

		GPU_CHECK_ERROR(cudaDestroyTextureObject(_preprocessedImageTexture));
		GPU_CHECK_ERROR(cudaDestroyTextureObject(_finalPyramidTexture));
		GPU_CHECK_ERROR(cudaDestroyTextureObject(_alphasTexture));
			
		GPU_CHECK_ERROR(cudaFree(_devOriginalImage));
		GPU_CHECK_ERROR(cudaFree(_devPreprocessedImage));
		GPU_CHECK_ERROR(cudaFree(_devPyramidImage));
		GPU_CHECK_ERROR(cudaFree(_devAlphaBuffer));
		GPU_CHECK_ERROR(cudaFree(_devDetections));
		GPU_CHECK_ERROR(cudaFree(_devDetectionCount));
		GPU_CHECK_ERROR(cudaFree(_devSurvivors));
		GPU_CHECK_ERROR(cudaFree(_devSurvivorCount));
	}

	void WaldboostDetector::setBlockSize(KernelType type, uint32 const& x, uint32 const& y, uint32 const& z)
	{
		_kernelBlockConfig[type] = dim3(x, y, z);
	}
}


