'''
This code evaluates the validation and test performance in an epoch of the model trained in jodie.py.
The task is: interaction prediction, i.e., predicting which action will a user interact with? 

To calculate the performance for one epoch:
$ python evaluate_interaction_prediction.py --network reddit --model jodie --epoch 49

To calculate the performance for all epochs, use the bash file, evaluate_all_epochs.sh, which calls this file once for every epoch.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

import argparse

from utilities import *
from data_loader import *
from models import *

# FIX THE RANDOM SEED FOR REPRODUCIBILITY
SEED = 1234
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False # reduces the performance

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Data file')
parser.add_argument('--model', default="mondrain_v0.1", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epoch', default=20, type=int, help='Epoch id to load')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')
parser.add_argument('--state_change', default=False, type=bool, help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 
args = parser.parse_args()
# Set the name of the data file 
args.dataname = args.data.split('.')[0]

# args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')
    
# SET GPU
args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# CHECK IF THE OUTPUT OF THE EPOCH IS ALREADY PROCESSED. IF SO, MOVE ON.
output_fname = "results/interaction_prediction_%s.txt" % args.dataname
if os.path.exists(output_fname):
    f = open(output_fname, "r")
    search_string = 'Test performance of epoch %d' % args.epoch
    for l in f:
        l = l.strip()
        if search_string in l:
            print("Output file already has results of epoch %d" % args.epoch)
            sys.exit(0)
    f.close()

# LOAD NETWORK
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

# SET TRAINING, VALIDATION AND TESTING SPLITS
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
test_end_idx = int(num_interactions * (args.train_proportion + 0.2))

# SET BATCHING TIMESPAN
'''
Timespan indicates how frequently the model is run and updated. 
All interactions in one timespan are processed simultaneously. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
At the end of each timespan, the model is updated as well. So, longer timespan means less frequent model updates. 
'''
# timespan = timestamp_sequence[-1] - timestamp_sequence[0]
# tbatch_timespan = timespan / 500 

# INITIALIZE MODEL PARAMETERS
model = Mondrian(args, num_users, num_actions, num_features).cuda()
model.cuda()
MSELoss = nn.MSELoss()

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# LOAD THE MODEL
model, optimizer, user_embeddings_dystat, action_embeddings_dystat, user_embeddings_timeseries, action_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch)
if train_end_idx != train_end_idx_training:
    sys.exit('Training proportion during training and testing are different. Aborting.')

# SET THE USER AND ACTION EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
set_embeddings_training_end(user_embeddings_dystat, action_embeddings_dystat, user_embeddings_timeseries, action_embeddings_timeseries, user_list_id, action_list_id, train_end_idx) 

# LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
action_embeddings = action_embeddings_dystat[:, :args.embedding_dim].detach()
# action_embeddings = action_embeddings.clone()
print(5*'*' + ' -- Memory allocated: ' + str(torch.cuda.max_memory_allocated()) + 5*'*')
action_embeddings_static = action_embeddings_dystat[:, args.embedding_dim:].detach()
# action_embeddings_static = action_embeddings_static.clone()
print(5*'*' + ' -- Memory allocated: ' + str(torch.cuda.max_memory_allocated()) + 5*'*')


user_embeddings = user_embeddings_dystat[:, :args.embedding_dim].detach()
# user_embeddings = user_embeddings.clone()
print(5*'*' + ' -- Memory allocated: ' + str(torch.cuda.max_memory_allocated()) + 5*'*')
user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:].detach()
# user_embeddings_static = user_embeddings_static.clone()
print(5*'*' + ' -- Memory allocated: ' + str(torch.cuda.max_memory_allocated()) + 5*'*')

# PERFORMANCE METRICS
validation_ranks = []
test_ranks = []

''' 
Here we use the trained model to make predictions for the validation and testing interactions.
The model does a forward pass from the start of validation till the end of testing.
For each interaction, the trained model is used to predict the embedding of the action it will interact with. 
This is used to calculate the rank of the true action the user actually interacts with.

After this prediction, the errors in the prediction are used to calculate the loss and update the model parameters. 
This simulates the real-time feedback about the predictions that the model gets when deployed in-the-wild. 
Please note that since each interaction in validation and test is only seen once during the forward pass, there is no data leakage. 
'''
tbatch_start_time = None
loss = 0
# FORWARD PASS
print("*** Making interaction predictions by forward pass (no t-batching) ***")
with trange(train_end_idx, test_end_idx) as progress_bar:
    for j in progress_bar:
        progress_bar.set_description('%dth interaction for validation and testing' % j)

        # LOAD INTERACTION J
        userid = user_list_id[j]
        actionid = action_list_id[j]
        feature = feature_list[j]
        user_timediff = user_timediference_list[j]
        action_timediff = action_timediference_list[j]
        # timestamp = timestamp_sequence[j]
        # if not tbatch_start_time:
        #     tbatch_start_time = timestamp
        actionid_previous = user_previous_actionid_list[j]

        # LOAD USER AND action EMBEDDING
        user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
        user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]
        action_embedding_input = action_embeddings[torch.cuda.LongTensor([actionid])]
        action_embedding_static_input = action_embeddings_static[torch.cuda.LongTensor([actionid])]
        feature_tensor = Variable(torch.Tensor(feature).cuda()).unsqueeze(0)
        user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).cuda()).unsqueeze(0)
        action_timediffs_tensor = Variable(torch.Tensor([action_timediff]).cuda()).unsqueeze(0)
        action_embedding_previous = action_embeddings[torch.cuda.LongTensor([actionid_previous])]

        # PROJECT USER EMBEDDING
        user_projected_embedding = model.forward(user_embedding_input, action_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
        user_action_embedding = torch.cat([user_projected_embedding, action_embedding_previous, action_embeddings_static[torch.cuda.LongTensor([actionid_previous])], user_embedding_static_input], dim=1)

        # PREDICT ACTION EMBEDDING
        predicted_action_embedding = model.predict_action_embedding(user_action_embedding)

        # CALCULATE PREDICTION LOSS
        loss += MSELoss(predicted_action_embedding, torch.cat([action_embedding_input, action_embedding_static_input], dim=1).detach())
        
        # CALCULATE DISTANCE OF PREDICTED ACTION EMBEDDING TO ALL actionS 
        euclidean_distances = nn.PairwiseDistance()(predicted_action_embedding.repeat(num_actions, 1), torch.cat([action_embeddings, action_embeddings_static], dim=1)).squeeze(-1) 
        
        # CALCULATE RANK OF THE TRUE ACTION AMONG ALL ACTIONS
        true_action_distance = euclidean_distances[actionid]
        euclidean_distances_smaller = (euclidean_distances < true_action_distance).data.cpu().numpy()
        true_action_rank = np.sum(euclidean_distances_smaller) + 1

        if j < test_start_idx:
            validation_ranks.append(true_action_rank)
        else:
            test_ranks.append(true_action_rank)

        # UPDATE USER AND action EMBEDDING
        user_embedding_output = model.forward(user_embedding_input, action_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update') 
        action_embedding_output = model.forward(user_embedding_input, action_embedding_input, timediffs=action_timediffs_tensor, features=feature_tensor, select='action_update') 

        # SAVE EMBEDDINGS
        action_embeddings[actionid,:] = action_embedding_output.squeeze(0) 
        user_embeddings[userid,:] = user_embedding_output.squeeze(0) 
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
        action_embeddings_timeseries[j, :] = action_embedding_output.squeeze(0)

        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
        loss += MSELoss(action_embedding_output, action_embedding_input.detach())
        loss += MSELoss(user_embedding_output, user_embedding_input.detach())

        # CALCULATE STATE CHANGE LOSS
        if args.state_change:
            loss += calculate_state_prediction_loss(model, [j], user_embeddings_timeseries, y_true, crossEntropyLoss) 

        # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
        # if timestamp - tbatch_start_time > tbatch_timespan:
        # tbatch_start_time = timestamp
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # RESET LOSS FOR NEXT T-BATCH
        loss = 0
        action_embeddings.detach_()
        user_embeddings.detach_()
        action_embeddings_timeseries.detach_() 
        user_embeddings_timeseries.detach_()
            
prepare_data_folder(args)
torch.save(user_embeddings, 'embeddings/%s/predicted_%s_embeddings.pt' % (args.dataname, args.dataname))
            
# CALCULATE THE PERFORMANCE METRICS
performance_dict = dict()
ranks = validation_ranks
mrr = np.mean([1.0 / r for r in ranks])
rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
performance_dict['validation'] = [mrr, rec10]

ranks = test_ranks
mrr = np.mean([1.0 / r for r in ranks])
rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)
performance_dict['test'] = [mrr, rec10]

# PRINT AND SAVE THE PERFORMANCE METRICS
fw = open(output_fname, "a")
metrics = ['Mean Reciprocal Rank', 'Recall@10']

print('\n\n*** Validation performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Validation performance of epoch %d ***\n' % args.epoch)
for i in range(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['validation'][i]))
    fw.write("Validation: " + metrics[i] + ': ' + str(performance_dict['validation'][i]) + "\n")
    
print('\n\n*** Test performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Test performance of epoch %d ***\n' % args.epoch)
for i in range(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['test'][i]))
    fw.write("Test: " + metrics[i] + ': ' + str(performance_dict['test'][i]) + "\n")

fw.flush()
fw.close()
