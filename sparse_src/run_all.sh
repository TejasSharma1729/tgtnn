#!/bin/bash
if [ $# -ne 6 ]; then
    echo "Usage: $0 <N> <Q> <D> <S> <DDPETH> <QDEPTH>"
    exit 1
fi
N=$1
Q=$2
D=$3
S=$4
DDEPTH=$5
QDEPTH=$6
dataset=random_${N}_${Q}_${D}_${S}
mkdir -p ${dataset}
if ! test -f ${dataset}/X.txt; then
    python3 get_random_dataset.py $N $Q $D $S ${dataset};
elif ! test -f ${dataset}/Q.txt; then
    python3 get_random_dataset.py $N $Q $D $S ${dataset};
fi
for ARG in PLAIN SMART CLASSWISE CLASSWISE_SMART; do
    g++ -std=c++17 -O3 -D${ARG} -DDDEPTH=${DDEPTH} -DQDEPTH=${QDEPTH} -march=native -mavx2 -I eigen-3.4.0/ -I ../assets/ -I json/include sparse_GTnn.cpp -o sparse_GTnn;
    nohup ./sparse_GTnn ${dataset} > ${dataset}/${ARG}_${DDEPTH}_${QDEPTH}.out &
done
