#!/bin/bash
make clean && make
echo;

echo "Testing on sparse-full dataset..."
python -u main.py -V MinHash -d sparseFull -H 100 -t 50 -s 500 -q 1000
echo;

echo "Testing on Movielens dataset..."
python -u main.py -V MinHash -d movielens -H 300 -t 10 -s 1000 -q 1000
echo;

echo "Testing on KDDB dataset..."
python -u main.py -V MinHash -d kddb -N 10000000 -H 100 -t 50 -s 1000 -q 1000
echo;

echo "Testing on Avazu-app dataset..."
python -u main.py -V WeightedMinHash -N 10000000 -d avazu -H 100 -t 50 -s 500 -q 1000
echo;

# Restricted avazu to 100 queries for 