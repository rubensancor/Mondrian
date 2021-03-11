import time
import argparse
import socket
from utilities import *
from models import *
import models as lib
from data_loader import *
import wandb

# FIX THE RANDOM SEED FOR REPRODUCIBILITY
SEED = 1234
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False # reduces the performance

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Data file')
parser.add_argument('--model', default="mondrain_v0.1", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=20, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('-ws', '--wandb_sync', '--wandb_sync=1', action='store_true', help='Check if the run is going to be uploaded to WandB')
parser.add_argument('--state_change', default=False, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
parser.add_argument('--full_train', default=False, action='store_true', help='Enables the full train mode training the model with all the dataset to calculate the embeddings in the final interaction.')
parser.add_argument('--tqdmdis', action='store_true', help='Enable or disable TQDM progress bar.')
parser.add_argument('--split', default=0, type=int, help='The split of the pipeline') 
args = parser.parse_args()

# Set the name of the data file 
args.dataname = args.data.split('.')[0]
tags = [socket.gethostname(),'JODIE_seq', args.dataname]
if not args.wandb_sync:
    os.environ['WANDB_MODE'] = 'dryrun'

if args.full_train:
    tags.append('full_train')

wandb.init(project="mondrian", config=args, tags=tags)

output_fname = "results/interaction_prediction_%s.txt" % args.data

# SET GPU
if args.gpu == -1:
    args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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
users_edited = [False] * num_users
print("*** Network statistics:\n  %d users\n  %d action types\n  %d features\n  %d interactions\n ***\n\n" % (num_users, num_actions, num_features, num_interactions))

# Calculate the weights assigned to each class to manage the imbalance
# weights = []
# counter = Counter(y_true)
# max_class = counter.most_common()[0][1]
# for i in counter:
#     weights.append(max_class/counter[i])
# y_true = torch.LongTensor(y_true)


# SET TRAINING, VALIDATION AND TESTING SPLITS
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
test_end_idx = int(num_interactions * (args.train_proportion + 0.2))

if args.full_train:
    end_idx = num_interactions
else:
    end_idx = train_end_idx

# INITIALIZE MODEL AND PARAMETERS
model = Mondrian(args, num_users, num_actions, num_features).cuda()
model.cuda()
MSELoss = nn.MSELoss()
# Weight assigned to each class
# weight = torch.Tensor(weights).cuda()
# crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
initial_action_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_action_embedding = initial_action_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
action_embeddings = initial_action_embedding.repeat(num_actions, 1)
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings
action_embedding_static = Variable(torch.eye(num_actions).cuda())

# PERFORMANCE METRICS
validation_ranks = []
test_ranks = []
mean_loss = []

# WANDB
wandb.watch(model)

# RUN THE MONDRAIN MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
AFTER EACH EPOCH THE MODEL IS TESTED WITH A VALIDATION SET TO EARLY STOP
'''
print("*** Training the MONDRIAN model for %d epochs ***" % args.epochs)

with trange(1, args.epochs+1, disable=args.tqdmdis) as progress_bar1:
    
    for ep in progress_bar1:
        progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))
        
        # print(5*'*' + ' START EPOCH -- Memory allocated: ' + str(torch.cuda.max_memory_allocated()) + 5*'*')
        epoch_start_time = time.time()
        
        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        action_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

        optimizer.zero_grad()
        total_loss, loss, total_interaction_count = 0, 0, 0
        

        model.train()

        with trange(end_idx, disable=args.tqdmdis) as progress_bar2:
            for j in progress_bar2:
                progress_bar2.set_description('Processed %dth interactions' % j)
                userid = user_list_id[j]
                actionid = action_list_id[j]
                feature = feature_list[j]
                previous_actionid = user_previous_actionid_list[j]
                
                users_edited[userid] = True

                feature_tensor = torch.Tensor(feature).cuda().unsqueeze(0)
                user_timediffs_tensor = torch.Tensor([user_timediference_list[j]]).cuda().unsqueeze(0)
                action_timediffs_tensor = torch.Tensor([action_timediference_list[j]]).cuda().unsqueeze(0)
                
                action_embedding_previous = action_embeddings[previous_actionid,:].unsqueeze(0)

                actual_user_embedding_static = user_embedding_static[userid,:].unsqueeze(0)
                previous_action_embedding_static = action_embedding_static[previous_actionid,:].unsqueeze(0)

                # PROJECT USER EMBEDDING TO CURRENT TIME
                user_embedding_input = user_embeddings[userid,:].unsqueeze(0)
                user_projected_embedding = model.forward(user_embedding_input, action_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                user_action_embedding = torch.cat([user_projected_embedding, action_embedding_previous, previous_action_embedding_static, actual_user_embedding_static], dim=1)

                # PREDICT NEXT ACTION EMBEDDING                            
                predicted_action_embedding = model.predict_action_embedding(user_action_embedding)
                
                # CALCULATE PREDICTION LOSS
                action_embedding_input = action_embeddings[actionid,:].unsqueeze(0)
                action_embedding_static_sq = action_embedding_static[actionid,:].unsqueeze(0)
                loss += MSELoss(predicted_action_embedding, torch.cat([action_embedding_input, action_embedding_static_sq], dim=1).detach())

                # UPDATE DYNAMIC EMBEDDINGS AFTER ACTION
                user_embedding_output = model.forward(user_embedding_input, action_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                action_embedding_output = model.forward(user_embedding_input, action_embedding_input, timediffs=action_timediffs_tensor, features=feature_tensor, select='action_update')

                # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                loss += MSELoss(action_embedding_output, action_embedding_input.detach())
                loss += MSELoss(user_embedding_output, user_embedding_input.detach())

                # loss.requires_grad = True
                # BACKPROPAGATE ERROR 
                total_loss += loss.item()
                mean_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                user_embeddings[userid,:] = user_embedding_output
                action_embeddings[actionid,:] = action_embedding_output

                user_embeddings_timeseries[j,:] = user_embedding_output
                action_embeddings_timeseries[j,:] = action_embedding_output

                loss = 0
                action_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                user_embeddings.detach_()
                
                action_embeddings_timeseries.detach_() 
                user_embeddings_timeseries.detach_()

        wandb.log({'epoch': ep,
                   'total_loss_mean': np.mean(mean_loss)})
        
        print(5*'*' + ' END TRAIN -- Memory allocated: ' + str(torch.cuda.max_memory_allocated()) + 5*'*')
        print("\nLast epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        
        #END OF ONE EPOCH
        print("\nTotal loss in this epoch = %f\n" % (total_loss))

        # UNTIL MEMORY MANAGEMENT
        user_embeddings_dystat = torch.cat([user_embeddings.cpu(), user_embedding_static.cpu()], dim=1)
        action_embeddings_dystat = torch.cat([action_embeddings.cpu(), action_embedding_static.cpu()], dim=1) 
        
        
        if args.full_train:
            prepare_data_folder(args)
            args.model = 'mondrain_full_v0.1'
            torch.save(user_embeddings_dystat, 'embeddings/%s/full_%s_embeddings_ep%s.pt' % (args.dataname, args.dataname, str(ep)))
        
        if not ep%5:
            save_model(model, optimizer, args, ep, user_embeddings_dystat, action_embeddings_dystat, train_end_idx, user_embeddings_timeseries, action_embeddings_timeseries)
            

        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        action_embeddings = initial_action_embedding.repeat(num_actions, 1)
        print(5*'*' + ' END EPOCH -- Memory allocated: ' + str(torch.cuda.max_memory_allocated()) + 5*'*')

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print("\n\n*** Training complete. Saving final model. ***\n\n")
save_model(model, optimizer, args, ep, user_embeddings_dystat, action_embeddings_dystat, train_end_idx, user_embeddings_timeseries, action_embeddings_timeseries)

