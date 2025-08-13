#!/usr/bin/bash

$CXX $CXXFLAGS -I ../assets -O3 naive_threaded.cpp -o naive_threaded 
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi
echo "imagenet"
./naive_threaded /mnt/HDD-2/harsh/datasets/imagenet/output
echo "imdb_wiki"
./naive_threaded /mnt/HDD-2/harsh/datasets/imdb_wiki/output
echo "insta_1m"
./naive_threaded /mnt/HDD-2/harsh/datasets/insta_1m/output
echo "mirflickr"
./naive_threaded /mnt/HDD-2/harsh/datasets/mirflickr/output
