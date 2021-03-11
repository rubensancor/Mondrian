#!/bin/bash

while getopts d:e:m: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
    esac
done

for i in {1..17};
do
    python3 pipeline_mondrian_full.py --data $dataset --epochs 5 --split "$i" --full_train
    python3 pipeline_mondrian_full.py --data $dataset --epochs 5 -ws --split "$i" 
done