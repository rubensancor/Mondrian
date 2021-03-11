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


parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Data file')
parser.add_argument('--split', default=1, type=int, help='The split of the pipeline') 
parser.add_argument('--train_proportion', default=0.66, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
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
users_edited = [False] * num_users
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
embeddings = torch.load('embeddings/%s/predicted_%s_embeddings.pt' % (args.dataname, args.dataname))
full_embeddings = torch.load('embeddings/%s/full_%s_embeddings_ep20.pt' % (args.dataname, args.dataname))


for i in range(train_end_idx):
    userid = user_list_id[i]
    users_edited[userid] = True

# print(head_labels)
# print(users_edited)
# print(embeddings)
X = []
y = []
x_test = []
y_test = []
for i in range(len(users_edited)):
    if users_edited[i]:
        X.append(full_embeddings[i].tolist()[0:128])
        # X.append(embeddings[i].tolist()[0:128])
        y.append(head_labels[i])
    else:
        x_test.append(embeddings[i].tolist())
        y_test.append(head_labels[i])

print(5 * '*' + ' Data distribution ' + 5 * '*')
print('Train: ')
print(collections.Counter(y))
print('Test: ')
print(collections.Counter(y_test))
print('\n')
print(collections.Counter(y)[0])
width = 0.35
ind = np.arange(2)
plot_0 = []
plot_1 = []
plot_0.append(collections.Counter(y)[0])
plot_0.append(collections.Counter(y_test)[0])
plot_1.append(collections.Counter(y)[1])
plot_1.append(collections.Counter(y_test)[1])
p1 = plt.bar(ind, plot_0, width)
p2 = plt.bar(ind, plot_1, width)

plt.ylabel('Quantity')
plt.title('User distribution')
plt.xticks(ind, ('Train', 'Test'))
plt.legend((p1[0], p2[0]), ('Malicious', 'Legit'))
plt.savefig('user_distribution.png')
exit()

oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)

 
# Linear SVC
clf = LinearSVC(class_weight='balanced')
clf.fit(X_over, y_over)
score = clf.score(x_test, y_test)
print('LinearSVC: %f' % score)


# KNN
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_over, y_over)
score = clf.score(x_test, y_test)
print('KNN: %f' % score)

# MLP
clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(X_over, y_over)
score = clf.score(x_test, y_test)
print('MLP: %f' % score)