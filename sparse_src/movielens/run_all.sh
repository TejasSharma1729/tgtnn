#!/bin/bash

for i in {00..15}; do
    file_name=movielens/shard_${i};
    outfile=movielens_knns/shard_${i}.txt;
    dataset=../data/${file_name}/X.csr;
    query_set=../data/${file_name}/Q.csr;
    ./optimized_gtnn ${dataset} ${query_set} > ${outfile};
done
