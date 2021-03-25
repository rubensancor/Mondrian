#!/bin/bash

while getopts d:e:m: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
    esac
done

python3 pipeline_mondrian_full.py --data $dataset --epochs 10 --full_train -ws --tags FULL_SPAIN
for i in {1..3};
do
    python3 pipeline_mondrian_full.py --data $dataset --epochs 10 -ws --split "$i" --tags FULL_SPAIN
done