#!/bin/bash
make clean && make
echo;

echo "Testing on sparse-full dataset..."
python -u test.py -V MinHash -d sparseFull -H 100 -t 50 -s 500 -q 1000
echo;

echo "Testing on Movielens dataset..."
python -u test.py -V MinHash -d movielens -H 300 -t 10 -s 1000 -q 1000
echo;

echo "Testing on KDDB dataset..."
python -u test.py -V MinHash -d kddb -H 100 -t 50 -s 1000 -q 1000
echo;

echo "Testing on Avazu-app dataset..."
python -u test.py -V MinHash -d avazu -H 100 -t 50 -s 500 -q 100
echo;

# Restricted avazu to 100 queries for 