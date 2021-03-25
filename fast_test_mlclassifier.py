import argparse
import collections
import socket
import wandb
from utilities import *
from data_loader import *
from models import *
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Data file')
parser.add_argument('-ws', '--wandb_sync', '--wandb_sync=1', action='store_true', help='Check if the run is going to be uploaded to WandB')
parser.add_argument('--split', default=0, type=float, help='The split of the pipeline') 
parser.add_argument('--epoch', default=20, type=int, help='Epoch id to load')
parser.add_argument('-t', '--tags', action='append', help='Tags for WandB')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
args = parser.parse_args()
args.dataname = args.data.split('.')[0]

tags = [socket.gethostname(),'FAST_TEST', args.dataname]
if args.tags is not None:
    tags = [socket.gethostname(), args.tags, args.dataname]
else:
    tags = [socket.gethostname(), args.dataname]
    
if not args.wandb_sync:
    os.environ['WANDB_MODE'] = 'dryrun'
    
wandb.init(project="mondrian", config=args, tags=tags)


# LOAD NETWORK
(user2id, user_list_id, user_timediference_list, user_previous_actionid_list,
 action2id, action_list_id, action_timediference_list, 
 timestamp_list,
 feature_list, 
 head_labels, 
 tail_labels) = load_data(args, tail_as_feature=True)

num_users = len(user2id)
num_interactions = len(user_list_id)
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
if args.split is not 0:
    print('*** LOADING EMBEDDINGS FROM SPLIT %s' % str(args.split))
    full_embeddings = torch.load('embeddings/%s/full_%s_embeddings_ep%s_split%s.pt' % (args.dataname, args.dataname, str(args.epoch), str(args.split)))
else:
    full_embeddings = torch.load('embeddings/%s/full_%s_embeddings_ep%s.pt' % (args.dataname, args.dataname, str(args.epoch)))


X = []
y = []


for i in range(len(full_embeddings)):
    X.append(full_embeddings[i].tolist()[0:128])
    y.append(head_labels[user_list_id.index(i)])

print(5 * '*' + ' Data distribution ' + 5 * '*')
print('Labels: ')
print(collections.Counter(y))

# width = 0.35
# ind = np.arange(2)
# plot_0 = []
# plot_1 = []
# plot_0.append(collections.Counter(y)[0])
# plot_0.append(collections.Counter(y_test)[0])
# plot_1.append(collections.Counter(y)[1])
# plot_1.append(collections.Counter(y_test)[1])
# p1 = plt.bar(ind, plot_0, width)
# p2 = plt.bar(ind, plot_1, width)

# plt.ylabel('Quantity')
# plt.title('User distribution')
# plt.xticks(ind, ('Train', 'Test'))
# plt.legend((p1[0], p2[0]), ('Malicious', 'Legit'))
# plt.savefig('user_distribution.png')

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y)

svc_f1 = []
svc_acc = []
knn_acc = []
knn_f1 = []
mlp_acc = []
mlp_f1 = []
rf_acc = []
rf_f1 = []
dummy_acc = []
dummy_f1 = []

for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = np.array(X)[train_index.astype(int)], np.array(X)[test_index.astype(int)]
    y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
 
    # Linear SVC
    clf = LinearSVC(class_weight='balanced')
    clf.fit(X, y)
    predictions = clf.predict(X_test)
    svc_acc = clf.score(X_test, y_test)
    svc_f1 = f1_score(y_test, predictions, average='macro')
    
    # KNN
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)
    predictions = clf.predict(X_test)
    knn_acc = clf.score(X_test, y_test)
    knn_f1 = f1_score(y_test, predictions, average='macro')
    
    # MLP
    clf = MLPClassifier(random_state=1, max_iter=300)
    clf.fit(X, y)
    predictions = clf.predict(X_test)
    mlp_acc = clf.score(X_test, y_test)
    mlp_f1 = f1_score(y_test, predictions, average='macro')
    
    # RF    
    clf = RandomForestClassifier()
    clf.fit(X, y)
    predictions = clf.predict(X_test)
    rf_acc = clf.score(X_test, y_test)
    rf_f1 = f1_score(y_test, predictions, average='macro')
    
    # DUMMY
    clf = DummyClassifier(strategy='constant', random_state=1234, constant=0)
    clf.fit(X, y)
    predictions = clf.predict(X_test)
    dummy_acc = clf.score(X_test, y_test)
    dummy_f1 = f1_score(y_test, predictions, average='macro')
   
wandb.log({'SVC_ACC': svc_acc,
           'SVC_F1': svc_f1,
           'KNN_ACC': knn_acc,
           'KNN_F1': knn_f1,
           'MLP_ACC': mlp_acc,
           'MLP_F1': mlp_f1,
           'RF_ACC': rf_acc,
           'RF_F1': rf_f1,
           'DUMMY_ACC': dummy_acc,
           'DUMMY_F1': dummy_f1})

