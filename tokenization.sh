#!/bin/bash

python3 pipeline_mondrian_full.py --data spain_users_interactions_hashtagtoken.csv --epochs 15 --split 1 --full_train --tags TOKENIZATION
python3 pipeline_mondrian_full.py --data spain_users_interactions_hashtagtoken.csv --epochs 15 -ws --split 1 --tags TOKENIZATION+tail
