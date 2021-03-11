import argparse

from utilities import *


parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True, help='Data path')
# parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="mondrain_v0.1", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_size', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
# parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
parser.add_argument('--draw_final', action='store_true', help='Enable or disable TQDM progress bar.')
args = parser.parse_args()

args.dataname = args.folder.split('/')[1]

for f in os.listdir(args.folder):
    if 'full' in f:
        state = 'Truth'
    else:
        state = 'Predicted'
        
    draw_embeddings(args, f, state)