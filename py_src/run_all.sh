#!/bin/bash

# Go to the directory py_src/
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

python -u mlgt.py --dataset imagenet --num_queries 1000 > results_imagenet.out
python -u mlgt.py --dataset imdb_wiki --num_queries 1000 > results_imdb_wiki.out
python -u mlgt.py --dataset insta_1m --num_queries 1000 > results_insta_1m.out
python -u mlgt.py --dataset mirflickr --num_queries 1000 > results_mirflickr.out
