import argparse
import collections
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from utilities import *
from data_loader import *
from models import *
from sklearn import svm
import imblearn
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Data file')
parser.add_argument('--split', default=0, type=int, help='The split of the pipeline') 
parser.add_argument('--epoch', default=20, type=int, help='Epoch id to load')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
args = parser.parse_args()
args.dataname = args.data.split('.')[0]

# LOAD NETWORK
(user2id, user_list_id, user_timediference_list, user_previous_actionid_list,
 action2id, action_list_id, action_timediference_list, 
 timestamp_list,
 feature_list, 
 head_labels, 
 tail_labels) = load_data(args)

num_users = len(user2id)
num_interactions = len(user_list_id)
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
full_embeddings = torch.load('embeddings/%s/full_%s_embeddings_ep%s.pt' % (args.dataname, args.dataname, str(args.epoch)))


# print(user_list_id.index(21323))
# print(head_labels[user_list_id.index(0)]) 
# exit()
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

svc = []
knn = []
mlp = []
rf = []
dummy = []

for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = np.array(X)[train_index.astype(int)], np.array(X)[test_index.astype(int)]
    y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]

 
    # Linear SVC
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    svc.append(score)
    


    # KNN
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    knn.append(score)
    

    # MLP
    clf = MLPClassifier(random_state=1, max_iter=300)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    mlp.append(score)
    
    # RF
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    rf.append(score)
    
    
    # Dummy
    clf = DummyClassifier(strategy='most_frequent')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    dummy.append(score)
   
print('LinearSVC: %f' % np.mean(svc)) 
print('KNN: %f' % np.mean(knn)) 
print('MLP: %f' % np.mean(mlp)) 
print('RF: %f' % np.mean(rf)) 
print('DUMMY: %f' % np.mean(dummy)) 

