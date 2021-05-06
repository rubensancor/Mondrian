import time
import argparse
from utilities import *
from models import *
import models as lib
from data_loader import *
import wandb
from statistics import mean
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

# FIX THE RANDOM SEED FOR REPRODUCIBILITY
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False # reduces the performance

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', required=True, help='Data file')
parser.add_argument('--model', default="mondrain_v0.1", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=512, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('-ws', '--wandb_sync', '--wandb_sync=1', action='store_true', help='Check if the run is going to be uploaded to WandB')
parser.add_argument('--state_change', default=False, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
parser.add_argument('--tail_as_feat', default=False, type=bool, help='Tail label as feature.')
parser.add_argument('--tqdmdis', action='store_true', help='Enable or disable TQDM progress bar.')
parser.add_argument('--data_split', required=True, type=float, help='Train/test split. Set the percentage for the training set.') 
# parser.add_argument('--test_size', required=True, type=float, help='Train/test split. Set the percentage for the training set.')
parser.add_argument('-n', '--name', help='Name of the run in WandB.')
parser.add_argument('-t', '--tags', action='append', help='Tags for WandB')
args = parser.parse_args()

# Set the name of the data file 
args.dataname = args.data.split('.')[0]

if '/' in args.dataname:
    args.dataname = args.dataname.split('/')[-1]
    
if args.tags is not None:
    tags = args.tags
    tags.append(args.dataname)
else:
    tags = args.dataname

if not args.wandb_sync:
    os.environ['WANDB_MODE'] = 'dryrun'

if args.name is not None:
    wandb.init(project="mondrian", name=args.name, config=args, tags=tags)
else:    
    wandb.init(project="mondrian", config=args, tags=tags)

# SET GPU
if args.gpu == -1:
    args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# LOAD DATA
(user2id, user_list_id, user_timediference_list, user_previous_actionid_list,
 action2id, action_list_id, action_timediference_list, 
 timestamp_list,
 feature_list, 
 head_labels, 
 tail_labels,
 reduced_head_list,
 tweet_list) = load_data(args, tail_as_feature=False)

num_users = len(user2id)
num_actions = len(action2id) + 1 # If the previous action is none
num_interactions = len(user_list_id)
num_features = len(feature_list[0])
y_true = action_list_id
edited_sentences = [0] * num_users
print("*** Network statistics:\n  %d users\n  %d action types\n  %d features\n  %d interactions\n ***\n\n" % (num_users, num_actions, num_features, num_interactions))

# sentence_embeddings = model.encode(tweet_list, batch_size=256 ,show_progress_bar=True)
user_embeddings = torch.empty(num_users, args.embedding_dim).cuda()
nn.init.xavier_uniform_(user_embeddings)

# SET TRAINING AND TESTING SPLITS
train_end_idx = int(num_interactions * args.data_split)

# PERFORMANCE METRICS
test_ranks = []
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

# Get edited users
edited_users = get_edited_users(args)

with trange(train_end_idx) as progress_bar1:
    for j in progress_bar1:
        userid = user_list_id[j]
        tweet_embedding = model.encode(tweet_list[j])
        tweet_embedding_tensor = torch.Tensor(tweet_embedding).cuda()
        
        # print(tweet_embedding_tensor.shape)
        # print(torch.cat((user_embeddings[userid],tweet_embedding_tensor),0))
        # tensortest = torch.cat((user_embeddings[userid],tweet_embedding_tensor), dim=1)
        # if True in torch.isnan(user_embeddings[userid]):
        #     print(userid)
        edited_sentences[userid] += 1
        user_embeddings[userid] = user_embeddings[userid].add_(tweet_embedding_tensor)
        
        

for userid in range(num_users):
    if edited_sentences[userid] > 1:
        user_embeddings[userid] = torch.div(user_embeddings[userid], edited_sentences[userid])
    
X = []
y = []
    
for i in range(len(user_embeddings)):
    if edited_users[i]:
        X.append(user_embeddings[i].tolist())
        y.append(head_labels[user_list_id.index(i)])
    
print(5 * '*' + ' Data distribution ' + 5 * '*')
print('Labels: ')
print(Counter(y))

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = np.array(X)[train_index.astype(int)], np.array(X)[test_index.astype(int)]
    y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]

    # Linear SVC
    clf = LinearSVC(class_weight='balanced')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    svc_acc.append(clf.score(X_test, y_test))
    svc_f1.append(f1_score(y_test, predictions, average='macro'))
    
    # KNN
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    knn_acc.append(clf.score(X_test, y_test))
    knn_f1.append(f1_score(y_test, predictions, average='macro'))
    
    # MLP
    clf = MLPClassifier(random_state=1, max_iter=300)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    mlp_acc.append(clf.score(X_test, y_test))
    mlp_f1.append(f1_score(y_test, predictions, average='macro'))
    
    # RF    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    rf_acc.append(clf.score(X_test, y_test))
    rf_f1.append(f1_score(y_test, predictions, average='macro'))
    
    print('\nSPLIT --> %i' % len(svc_acc))
    user_guessed = []
    for index, i in enumerate(predictions):
        if i == y_test[index]:
            user_guessed.append(True)
        else:
            user_guessed.append(False)
    
    
    # DUMMY
    clf = DummyClassifier(strategy='stratified', random_state=1234)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    dummy_acc.append(clf.score(X_test, y_test))
    dummy_f1.append(f1_score(y_test, predictions, average='macro'))

wandb.log({'SVC_ACC': mean(svc_acc),
        'SVC_F1': mean(svc_f1),
        'KNN_ACC': mean(knn_acc),
        'KNN_F1': mean(knn_f1),
        'MLP_ACC': mean(mlp_acc),
        'MLP_F1': mean(mlp_f1),
        'RF_ACC': mean(rf_acc),
        'RF_F1': mean(rf_f1),
        'DUMMY_ACC': mean(dummy_acc),
        'DUMMY_F1': mean(dummy_f1)})

