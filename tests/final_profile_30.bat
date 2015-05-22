cd ../bin
nvprof --log-file ../out/prof_ashared_8x8_1080.csv --csv --kernels "::detect:" --metrics achieved_occupancy,shared_replay_overhead,shared_load_throughput,shared_store_throughput WaldboostDetector_30.exe -iv 1920x1080.mp4 -bs 8 -dm ashared -lf 0
nvprof --log-file ../out/prof_ashared_16x16_1080.csv --csv --kernels "::detect:" --metrics achieved_occupancy,shared_replay_overhead,shared_load_throughput,shared_store_throughput WaldboostDetector_30.exe -iv 1920x1080.mp4 -bs 16 -dm ashared -lf 0
nvprof --log-file ../out/prof_ashared_32x32_1080.csv --csv --kernels "::detect:" --metrics achieved_occupancy,shared_replay_overhead,shared_load_throughput,shared_store_throughput WaldboostDetector_30.exe -iv 1920x1080.mp4 -bs 32 -dm ashared -lf 0

nvprof --log-file ../out/prof_ashared_8x8_480.csv --csv --kernels "::detect:" --metrics achieved_occupancy,shared_replay_overhead,shared_load_throughput,shared_store_throughput WaldboostDetector_30.exe -iv 720x480.mp4 -bs 8 -dm ashared -lf 0
nvprof --log-file ../out/prof_ashared_16x16_480.csv --csv --kernels "::detect:" --metrics achieved_occupancy,shared_replay_overhead,shared_load_throughput,shared_store_throughput WaldboostDetector_30.exe -iv 720x480.mp4 -bs 16 -dm ashared -lf 0
nvprof --log-file ../out/prof_ashared_32x32_480.csv --csv --kernels "::detect:" --metrics achieved_occupancy,shared_replay_overhead,shared_load_throughput,shared_store_throughput WaldboostDetector_30.exe -iv 720x480.mp4 -bs 32 -dm ashared -lf 0
pause