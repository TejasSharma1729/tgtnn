#!/usr/bin/bash

$CXX $CXXFLAGS -I ../assets -O3 optimized_gtnn.cpp -o optimized_gtnn 
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi
echo "Compilation successful. Running the program..."
# ./optimized_gtnn /mnt/HDD-2/harsh/datasets/imagenet/output > results/imagenet.txt
# ./optimized_gtnn /mnt/HDD-2/harsh/datasets/imdb_wiki/output > results/imdb_wiki.txt
# ./optimized_gtnn /mnt/HDD-2/harsh/datasets/insta_1m/output > results/insta_1m.txt
# ./optimized_gtnn /mnt/HDD-2/harsh/datasets/mirflickr/output > results/mirflickr.txt
echo "imagenet"
./optimized_gtnn /mnt/HDD-2/harsh/datasets/imagenet/output
echo "imdb_wiki"
./optimized_gtnn /mnt/HDD-2/harsh/datasets/imdb_wiki/output
echo "insta_1m"
./optimized_gtnn /mnt/HDD-2/harsh/datasets/insta_1m/output
echo "mirflickr"
./optimized_gtnn /mnt/HDD-2/harsh/datasets/mirflickr/output
