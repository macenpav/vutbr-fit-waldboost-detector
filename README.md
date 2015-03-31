# WaldboostDetector

WaldboostDetector is a master's thesis project on <a href="http://www.fit.vutbr.cz/">BUT FIT</a> researching the topic of GPU accelerated object detection.

## What's inside the package?

* A binary detector with multiple detection options (both gpu and cpu)
* Source code + CMake build system

## How to build?

WaldboostDetector is only available as a binary built from the source code, but it's all pretty simple. 

* Download and install <a href="http://www.cmake.org/">CMake</a> if you haven't done so already.
* Clone the repository using the following command:
``` bash
$ git clone https://github.com/mmaci/vutbr-fit-waldboost-detector
```
* For CMake, the source code is found in the src folder. Binaries can be built wherever you want.
* Following dependencies are needed: *CUDA 5.0+*, *OpenCV 2.4.9+*

## Other information

The whole project is currently under construction. The implementation itself is mostly finished for the purpose of the thesis and I'm currently working on performance
measurements. If you are interested in this topic, would like to contribute or have anything else on mind, you can contact me at macenauer.p@gmail.com.
