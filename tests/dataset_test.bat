cd ../bin/images
dir *.pgm /b /s >> "../files.txt"
cd ..
WaldboostDetector.exe -id files.txt -oc test.csv -vd -bs 16 -v -dm ashared -t -lf 30 > "../tests/out_dataset.txt"
pause