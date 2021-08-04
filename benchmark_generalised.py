import os
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', required=True, help='Data file')
parser.add_argument('--train_size', required=True, type=int, help='Train/test split. Set the percentage for the training set.')
parser.add_argument('--test_size', type=float, help='Train/test split. Set the percentage for the training set.')
parser.add_argument('-t', '--tags', action='append', help='Tags for WandB')
parser.add_argument('-n', '--name', help='Name of the run in WandB.')
parser.add_argument('-s', '--skip', default=[], action='append', choices=['train', 'test', 'bert'], help='Use this flag if you want to skip some phases')
args = parser.parse_args()

range_end = 11
    
if 'train' not in args.skip:
    # Train 
    bashCmd = 'python3 generalised_mondrian.py --data ' + args.data + ' --train_split ' + str(args.train_size) + ' -t train -ws -n TRAIN_' + args.name
    
    subprocess.call(bashCmd, shell=True)

if 'test' not in args.skip:
    # Test
    
    for i in range(1, range_end):
        bashCmd = 'python3 generalised_evaluation.py --data ' + args.data + ' --train_split ' + str(args.train_size) + ' --test_size 0.' + str(i) + ' -t test -t test_0.' + str(i) + ' -ws -n TEST_' + args.name + '_0.' + str(i)
        if args.tags is not None:
            for tag in args.tags:
                bashCmd += ' -t ' + tag

        subprocess.call(bashCmd, shell=True)

if 'bert' not in args.skip:
    # Bert 
    for i in range(1, range_end):
        bashCmd = 'python3 bert_encoder.py --data ' + args.data + ' --train_split ' +  str(args.train_size) + ' --test_size 0.' + str(i) + ' -t test_0.' + str(i) + ' -t bert -ws -n BERT_' + args.name + '_0.' + str(i)
        if args.tags is not None:
            for tag in args.tags:
                bashCmd += ' -t ' + tag

        subprocess.call(bashCmd, shell=True)