![WBD](logo.png?raw=true)

## What’s WBD about?
Object detection is a process of finding real-world objects such as faces, vehicles or pedestrians in images and videos. WBD is a master's thesis project on <a href="http://www.fit.vutbr.cz/">BUT FIT</a> researching the topic of GPU accelerated object detection. The goal is to detect such objects, exploiting the capabilities of NVIDIA GPUs to their maximum and use them for other applications such as tracking, recognition or analysis. 

## What's inside the package?

* A binary detector with multiple detection options (both gpu and cpu)
* Source code + CMake build system
* Lots of documentation, such as the thesis itself

## How to build?

WaldboostDetector is only available as a binary built from the source code, but it's all pretty simple. 

* Download and install <a href="http://www.cmake.org/">CMake</a> if you haven't done so already.
* Clone the repository using the following command:
``` bash
$ git clone https://github.com/mmaci/vutbr-fit-waldboost-detector
```
* For CMake, the source code is found in the src folder. Binaries can be built wherever you want.
* Following dependencies are needed: *CUDA 5.0+*, *OpenCV 2.4.9+*

## Command-line tool description

* **-D --dataset [filename]** Input textfile containing dataset filenames.
* **-V --video [filename]** Input video.
* **-c --csv** CSV output.
* **-v --verbose** Verbose output.
* **-t --timer** Measure detection time.
* **-o --visualoutput** Shows a visual output.
* **-s --survivors** Outputs surviving threads every stage of the classifier.
* **-d --visualdebug** Shows a visual output including pyramidal and preprocessed image.
* **-b --blocksize [8/16/32]** Sets block size. Should be set to 8, 16 or 32 for (64, 256 or 1024 threads)
* **-l --limitframes [number]** Process only a given number of frames.
* **-m --detectionmode [global/shared/prefixsum/hybrid/cpu]** Sets the specific implementation to use.

## Other information

The whole project is currently under construction. The implementation itself is mostly finished for the purpose of the thesis and I'm currently working on performance
measurements. If you are interested in this topic, would like to contribute or have anything else on mind, you can contact me at macenauer.p@gmail.com.
