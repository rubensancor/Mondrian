import os
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', required=True, help='Data file')
parser.add_argument('--mode', required=True, choices=['train', 'test', 'bert'])
parser.add_argument('--test_size', type=float, help='Train/test split. Set the percentage for the training set.')
parser.add_argument('-ws', '--wandb_sync', '--wandb_sync=1', action='store_true', help='Check if the run is going to be uploaded to WandB')
parser.add_argument('-t', '--tags', action='append', help='Tags for WandB')



# parser.add_argument('--train_split', type=float, help='Train split. Set the percentage for the training set.')
# parser.add_argument('--data_split', type=float, help='Train/test split. Set the percentage for the training set.') 
# parser.add_argument('--model', default="mondrain_v0.1", help='Model name to save output in file')
# parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
# parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train the model')
# parser.add_argument('--embedding_dim', default=512, type=int, help='Number of dimensions of the dynamic embedding')
# parser.add_argument('--state_change', default=False, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
# parser.add_argument('--tail_as_feat', default=False, type=bool, help='Tail label as feature.')
# parser.add_argument('--tqdmdis', action='store_true', help='Enable or disable TQDM progress bar.')
# parser.add_argument('--data_split', required=True, type=float, help='Train/test split. Set the percentage for the training set.') 
# parser.add_argument('--name', help='Name of the run in WandB.')
args = parser.parse_args()


for i in range(1, 9):
    bashCmd = 'python3 '
    # Add program mode
    if args.mode == 'train':
        if args.train_split is not None:
            bashCmd += 'mondrian_sequential.py --data ' + args.data + '--train_split 0.' + str(i) + ' -t train'
        else:
            raise ValueError
    elif args.mode == 'test':
        if args.train_split is not None and args.test_size is not None:
            bashCmd += 'evaluate_mondrian.py --data ' + args.data + '--train_split 0.' + str(i) + ' --test_size ' + str(args.test_size) + ' -t test'
        else:
            raise ValueError
    elif args.mode == 'bert':
        bashCmd += 'bert_encoder.py --data ' + args.data + '--data_split 0.' + str(i) + ' -t bert'


    if args.tags is not None:
        for i in args.tags:
            bashCmd += ' -t ' + i

    if args.wandb_sync:
        bashCmd += ' -ws'

    print(bashCmd)
    subprocess.call(bashCmd, shell=True)

# process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)

# output, error = process.communicate()
