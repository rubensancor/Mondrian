import time
import argparse
from sklearn.decomposition import PCA
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw


from utilities import *
from models import *
from data_loader import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Data path')
# parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="mondrain_v0.1", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
# parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
parser.add_argument('--draw_final', action='store_true', help='Enable or disable TQDM progress bar.')
args = parser.parse_args()

# LOAD DATA
(user2id, user_list_id, user_timediference_list, user_previous_actionid_list,
 action2id, action_list_id, action_timediference_list, 
 timestamp_list,
 feature_list, 
 head_labels, 
 tail_labels) = load_data(args)
num_users = len(user2id)
num_actions = len(action2id) + 1 # If the previous action is none
num_interactions = len(user_list_id)
num_features = len(feature_list[0])
y_true = action_list_id
print("*** Network statistics:\n  %d users\n  %d action types\n  %d features\n  %d interactions\n ***\n\n" % (num_users, num_actions, num_features, num_interactions))

model = Mondrian(args, num_users, num_actions, num_features).cuda()
model.cuda()
# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epochs)


user_timeseries_tensor = torch.load('user_embeddings_timeseries.pt')
# user_timeseries_tensor = user_timeseries_tensor.cpu().numpy()
user_timeseries = user_embeddings_timeseries.cpu().numpy()


draw_final_embeddings(args.data.split('.')[0], user_embeddings_dystat, user_list_id)
exit()
# pca = PCA(n_components=3)
# user_timeseries_pca = pca.fit_transform(user_timeseries)

user_lines = {}


for user in user_list_id:
    user_lines[user] = []
for interaction in tqdm(range(num_interactions)):
    for user in user_lines:
        if user == user_list_id[interaction]:
            user_lines[user].append(user_timeseries[interaction])
        elif len(user_lines[user]) > 0:
            user_lines[user].append(user_lines[user][-1])
        else:
            user_lines[user].append([0 for i in range(128)])

features = []
for user in tqdm(user_lines):
    features.append(user_lines[user])

labels = []
for cnt, user in enumerate(tqdm(user_list_id)):
    if user == len(labels):
        labels.append(head_labels[cnt])

feature_tensor = torch.tensor(features)
feature_tensor = feature_tensor.view(num_users, num_interactions, 128)
label_tensor = torch.tensor(labels)
torch.save(feature_tensor, args.data.split('.')[0] + '_features.pt')
torch.save(label_tensor, args.data.split('.')[0] + '_labels.pt')
exit()

for cnt, user in enumerate(tqdm(user_lines)):
    if cnt < 0.8 * num_users:
        train.append(user_lines[user])
        train_labels.append(head_labels[user])
    else:
        test.append(user_lines[user])
        test_labels.append(head_labels[user])
    
# print('To array')
# train = np.fromiter(train, dtype=np.float32, count=400000)
# print('1/4')
# train_labels = np.asarray(train_labels)
# print('2/4')
# test = np.asarray(test)
# print('3/4')
# test_labels = np.asarray(test_labels)
# print('4/4')
# print('Escribiendo')
np.save('train_features.npy', train)
print('1/4')
np.save('train_labels.npy', train_labels)
print('2/4')
np.save('test_features.npy', test)
print('3/4')
np.save('test_labels.npy', train_labels)
print('4/4')
exit()
for user in tqdm(user_lines):
    x = np.array(user_lines[user]) 
    for user_2 in user_lines:
        if user == user_2:
            continue
        y = np.array(user_lines[user_2])
        distance, path = fastdtw(x, y, dist=euclidean)
        print('Labels: %i - %i' % (head_labels[user], head_labels[user_2]))        
        print('Distance between %i and %i: %f ' % (user, user_2, distance))
exit()


# pca = PCA(n_components=3)
# user_timeseries_pca = pca.fit_transform(user_timeseries)

# user_lines = dict((n, {}) for n in user_list_id)
# for cnt, embedding in enumerate(user_timeseries_pca):
#     print(embedding)
#     exit()
#     user_lines[user_list_id[cnt]][cnt] = embedding


# for user in user_lines:
#     x = []
#     y = []
#     for i in range(num_interactions):
#         last = [0,0,0]
#         if i in user_lines[user]:
#             x.append(user_lines[user][i][0])
#             y.append(user_lines[user][i][1])
#             z.append(user_lines[user][i][2])
#             last = user_lines[user][i]
#         else:
#             x.append(last[0])
#             y.append(last[1])
#             z.append(last[2])

