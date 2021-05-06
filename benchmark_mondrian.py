import os
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', required=True, help='Data file')
parser.add_argument('--test_size', required=True, type=float, help='Train/test split. Set the percentage for the training set.')
parser.add_argument('-t', '--tags', action='append', help='Tags for WandB')
parser.add_argument('-n', '--name', help='Name of the run in WandB.')
parser.add_argument('-s', '--skip', default=[], action='append', choices=['train', 'test', 'bert'], help='Use this flag if you want to skip some phases')
args = parser.parse_args()

if 'train' not in args.skip:
    # Train 
    for i in range(1, 11):
        if i == 10:
            bashCmd = 'python3 mondrian_sequential.py --data ' + args.data + ' --train_split 1 -t train -ws -n TRAIN_' + args.name + '_1'
        else:
            bashCmd = 'python3 mondrian_sequential.py --data ' + args.data + ' --train_split 0.' + str(i) + ' -t train -ws -n TRAIN_' + args.name + '_0.' + str(i)
        if args.tags is not None:
            for tag in args.tags:
                bashCmd += ' -t ' + tag

        subprocess.call(bashCmd, shell=True)

if 'test' not in args.skip:
    # Test 
    for i in range(1, 11):
        if i == 10:
            bashCmd = 'python3 evaluate_mondrian.py --data ' + args.data + ' --train_split 1 --test_size ' + str(args.test_size) + ' -t test -t test_' + str(args.test_size) + ' -ws -n TEST_' + args.name + '_1'
        else:
            bashCmd = 'python3 evaluate_mondrian.py --data ' + args.data + ' --train_split 0.' + str(i) + ' --test_size ' + str(args.test_size) + ' -t test -t test_' + str(args.test_size) + ' -ws -n TEST_' + args.name + '_0.' + str(i)
        
        if args.tags is not None:
            for tag in args.tags:
                bashCmd += ' -t ' + tag

        subprocess.call(bashCmd, shell=True)

if 'bert' not in args.skip:
    # Bert 
    for i in range(1, 11):
        if i == 10:
            bashCmd = 'python3 bert_encoder.py --data ' + args.data + ' --data_split 1 -t bert -ws -n BERT_' + args.name + '_1'
        else:
            bashCmd = 'python3 bert_encoder.py --data ' + args.data + ' --data_split 0.' + str(i) + ' -t bert -ws -n BERT_' + args.name + '_0.' + str(i)
        if args.tags is not None:
            for tag in args.tags:
                bashCmd += ' -t ' + tag

        subprocess.call(bashCmd, shell=True)