#!/bin/bash
make clean && make
echo;

echo "Testing on sparse-full dataset..."
# python -u main.py -V MinHash -d sparseFull -H 100 -t 50 -s 500 -q 1000
PYTHONFAULTHANDLER=1 python3 main.py --dataset sparseFull --variants "MinHash" --num-neighbors 10 --num-hashes 100 --threshold 50 --target-sparsity 500 --num-queries 1000
echo;

echo "Testing on Movielens dataset..."
# python -u main.py -V MinHash -d movielens -H 300 -t 10 -s 1000 -q 1000
PYTHONFAULTHANDLER=1 python3 main.py --dataset movielens --variants "MinHash" --num-neighbors 10 --num-hashes 300 --threshold 10 --target-sparsity 1000 --num-queries 1000
echo;

echo "Testing on KDDB dataset..."
# python -u main.py -V MinHash -d kddb -n 10000000 -H 100 -t 50 -s 1000 -q 1000
PYTHONFAULTHANDLER=1 python3 main.py --dataset kddb --variants "MinHash" --num-neighbors 10 --num-hashes 100 --threshold 50 --target-sparsity 1000 --num-features 10000000 --num-queries 1000
echo;

echo "Testing on Avazu-app dataset..."
# python -u main.py -V WeightedMinHash -n 10000000 -d avazu -H 100 -t 50 -s 500 -q 1000
PYTHONFAULTHANDLER=1 python3 main.py --dataset avazu --variants "WeightedMinHash" --num-neighbors 10 --num-hashes 100 --threshold 50 --target-sparsity 500 --num-features 10000000 --num-queries 1000
echo;

# Restricted avazu to 100 queries for 