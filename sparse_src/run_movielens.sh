#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_movielens.sh <DDEPTH> <QDEPTH>";
    exit 1;
fi
DDEPTH=$1
QDEPTH=$2
for i in {00..15}; do
    if ! test -f movielens/shard_${i}/X.txt; then
        python3 ./read_movielens.py ${i};
    elif ! test -f movielens/shard_${i}/Q.txt; then
        python3 ./read_movielens.py ${i};
    fi
    dataset=movielens/shard_${i};
    children=(0 0 0 0);
    ARGS=(PLAIN SMART CLASSWISE CLASSWISE_SMART);
    algs=(sparse_plain sparse_smart sparse_classwise sparse_classwise_smart);
    for j in {0..3}; do
        g++ -std=c++17 -O3 -D${ARGS[${j}]} -DDDEPTH=${DDEPTH} -DQDEPTH=${QDEPTH} sparse_GTnn.cpp -o sparse_GTnn;
        outfile=${dataset}/${algs[${j}]}_${DDEPTH}_${QDEPTH}.out;
        taskset -c $(( ${j} + 7 )) ./sparse_GTnn ${dataset} > ${outfile} & children[${j}]=$!;
    done
    for j in {0..3}; do
        wait ${children[${j}]};
    done
    rm -rf movielens/shard_${i}/X.txt movielens/shard_${i}/Q.txt;
done
python3 movielens/overall/compute_average.py ${DDEPTH} ${QDEPTH};
