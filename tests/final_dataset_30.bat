cd ../bin/images
dir *.pgm /b /s >> "../files.txt"
cd ..
WaldboostDetector_30.exe -id files.txt -oc aglobal_8_dataset.csv -bs 8 -dm aglobal -t
WaldboostDetector_30.exe -id files.txt -oc aglobal_16_dataset.csv -bs 16 -dm aglobal -t
WaldboostDetector_30.exe -id files.txt -oc aglobal_32_dataset.csv -bs 32 -dm aglobal -t

WaldboostDetector_30.exe -id files.txt -oc ashared_8_dataset.csv -bs 8 -dm ashared -t
WaldboostDetector_30.exe -id files.txt -oc ashared_16_dataset.csv -bs 16 -dm ashared -t
WaldboostDetector_30.exe -id files.txt -oc ashared_32_dataset.csv -bs 32 -dm ashared -t

WaldboostDetector_30.exe -id files.txt -oc hybridsg_8_dataset.csv -bs 8 -dm hybridsg -t
WaldboostDetector_30.exe -id files.txt -oc hybridsg_16_dataset.csv -bs 16 -dm hybridsg -t
WaldboostDetector_30.exe -id files.txt -oc hybridsg_32_dataset.csv -bs 32 -dm hybridsg -t
pause