import time
import argparse

from utilities import *
from models import *
import models as lib
from data_loader import *
import wandb




parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Data path')
# parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="mondrain_v0.1", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('-ws', '--wandb_sync', '--wandb_sync=1', action='store_true', help='Check if the run is going to be uploaded to WandB')
parser.add_argument('--state_change', default=False, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
args = parser.parse_args()

if not args.wandb_sync:
    os.environ['WANDB_MODE'] = 'dryrun'

wandb.init(project="mondrian", config=args, tags='JODIE')

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
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the JODIE model is trained. 
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm). 
The batches are then used to train JODIE. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates. 
'''
timespan = timestamp_list[-1] - timestamp_list[0]
tbatch_timespan = timespan / 500
#Estaba en 500 antes

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

# WANDB
wandb.watch(model)

# RUN THE MONDRAIN MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
AFTER EACH EPOCH THE MODEL IS TESTED WITH A VALIDATION SET TO EARLY STOP
'''
print("*** Training the MONDRIAN model for %d epochs ***" % args.epochs)
is_first_epoch = True
cached_tbatches_user = {}
cached_tbatches_action = {}
cached_tbatches_interactionids = {}
cached_tbatches_feature = {}
cached_tbatches_user_timediffs = {}
cached_tbatches_action_timediffs = {}
cached_tbatches_previous_action = {}


with trange(1, args.epochs+1) as progress_bar1:
    
    for ep in progress_bar1:
        progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))
        
        epoch_start_time = time.time()
        
        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        action_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0
        
        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        model.train()

        with trange(train_end_idx) as progress_bar2:
            for j in progress_bar2:
                progress_bar2.set_description('Processed %dth interactions' % j)

                if is_first_epoch:
                    # READ INTERACTION J
                    userid = user_list_id[j]
                    actionid = action_list_id[j]
                    feature = feature_list[j]
                    user_timediff = user_timediference_list[j]
                    action_timediff = action_timediference_list[j]

                    # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                    tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_action[actionid]) + 1
                    lib.tbatchid_user[userid] = tbatch_to_insert
                    lib.tbatchid_action[actionid] = tbatch_to_insert

                    lib.current_tbatches_user[tbatch_to_insert].append(userid)
                    lib.current_tbatches_action[tbatch_to_insert].append(actionid)
                    lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                    lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                    lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                    lib.current_tbatches_action_timediffs[tbatch_to_insert].append(action_timediff)
                    lib.current_tbatches_previous_action[tbatch_to_insert].append(user_previous_actionid_list[j])

                timestamp = timestamp_list[j]
                if tbatch_start_time is None:
                    tbatch_start_time = timestamp

                # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
                if timestamp - tbatch_start_time > tbatch_timespan:
                    tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

                    # ITERATE OVER ALL T-BATCHES
                    if not is_first_epoch:
                        lib.current_tbatches_user = cached_tbatches_user[timestamp]
                        lib.current_tbatches_action = cached_tbatches_action[timestamp]
                        lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                        lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                        lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                        lib.current_tbatches_action_timediffs = cached_tbatches_action_timediffs[timestamp]
                        lib.current_tbatches_previous_action = cached_tbatches_previous_action[timestamp]

                    with trange(len(lib.current_tbatches_user)) as progress_bar3:
                        for i in progress_bar3:
                            progress_bar3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))
                            
                            total_interaction_count += len(lib.current_tbatches_interactionids[i])

                            # LOAD THE CURRENT TBATCH
                            if is_first_epoch:
                                lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).cuda()
                                lib.current_tbatches_action[i] = torch.LongTensor(lib.current_tbatches_action[i]).cuda()
                                lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                                lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).cuda()

                                lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()
                                lib.current_tbatches_action_timediffs[i] = torch.Tensor(lib.current_tbatches_action_timediffs[i]).cuda()
                                lib.current_tbatches_previous_action[i] = torch.LongTensor(lib.current_tbatches_previous_action[i]).cuda()

                            tbatch_userids = lib.current_tbatches_user[i] # Recall "lib.current_tbatches_user[i]" has unique elements
                            tbatch_actionids = lib.current_tbatches_action[i] # Recall "lib.current_tbatches_item[i]" has unique elements
                            tbatch_interactionids = lib.current_tbatches_interactionids[i]
                            feature_tensor = Variable(lib.current_tbatches_feature[i]) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                            user_timediffs_tensor = Variable(lib.current_tbatches_user_timediffs[i]).unsqueeze(1)
                            action_timediffs_tensor = Variable(lib.current_tbatches_action_timediffs[i]).unsqueeze(1)
                            tbatch_actionids_previous = lib.current_tbatches_previous_action[i]
                            action_embedding_previous = action_embeddings[tbatch_actionids_previous,:]

                            # PROJECT USER EMBEDDING TO CURRENT TIME
                            user_embedding_input = user_embeddings[tbatch_userids,:]
                            user_projected_embedding = model.forward(user_embedding_input, action_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                            user_action_embedding = torch.cat([user_projected_embedding, action_embedding_previous, action_embedding_static[tbatch_actionids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)
                            
                            # PREDICT NEXT ACTION EMBEDDING                            
                            predicted_action_embedding = model.predict_action_embedding(user_action_embedding)

                            # CALCULATE PREDICTION LOSS
                            action_embedding_input = action_embeddings[tbatch_actionids,:]
                            loss += MSELoss(predicted_action_embedding, torch.cat([action_embedding_input, action_embedding_static[tbatch_actionids,:]], dim=1).detach())

                            # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                            user_embedding_output = model.forward(user_embedding_input, action_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                            action_embedding_output = model.forward(user_embedding_input, action_embedding_input, timediffs=action_timediffs_tensor, features=feature_tensor, select='action_update')

                            action_embeddings[tbatch_actionids,:] = action_embedding_output
                            user_embeddings[tbatch_userids,:] = user_embedding_output  

                            user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                            action_embeddings_timeseries[tbatch_interactionids,:] = action_embedding_output

                            # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                            loss += MSELoss(action_embedding_output, action_embedding_input.detach())
                            loss += MSELoss(user_embedding_output, user_embedding_input.detach())

                            # CALCULATE STATE CHANGE LOSS
                            if args.state_change:
                                loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss) 
                            
                    # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # RESET LOSS FOR NEXT T-BATCH
                    loss = 0
                    action_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                    user_embeddings.detach_()
                    action_embeddings_timeseries.detach_() 
                    user_embeddings_timeseries.detach_()
                   
                    # REINITIALIZE
                    if is_first_epoch:
                        cached_tbatches_user[timestamp] = lib.current_tbatches_user
                        cached_tbatches_action[timestamp] = lib.current_tbatches_action
                        cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                        cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                        cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                        cached_tbatches_action_timediffs[timestamp] = lib.current_tbatches_action_timediffs
                        cached_tbatches_previous_action[timestamp] = lib.current_tbatches_previous_action
                        
                        reinitialize_tbatches()
                        tbatch_to_insert = -1

        is_first_epoch = False # as first epoch ends here
        print("Last epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        # END OF ONE EPOCH 
        print("\n\nTotal loss in this epoch = %f" % (total_loss))
        wandb.log({'epoch': ep,
                   'loss': total_loss,
                   'time': (time.time()-epoch_start_time)/60 })

        action_embeddings_dystat = torch.cat([action_embeddings, action_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
        if not ep%5:
            save_model(model, optimizer, args, ep, user_embeddings_dystat, action_embeddings_dystat, train_end_idx, user_embeddings_timeseries, action_embeddings_timeseries)

        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        action_embeddings = initial_action_embedding.repeat(num_actions, 1)

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print("\n\n*** Training complete. Saving final model. ***\n\n")
save_model(model, optimizer, args, ep, user_embeddings_dystat, action_embeddings_dystat, train_end_idx, user_embeddings_timeseries, action_embeddings_timeseries)





#                 userid = user_list_id[j]
#                 actionid = action_list_id[j]
#                 feature = feature_list[j]
#                 previous_actionid = user_previous_actionid_list[j]

#                 feature_tensor = torch.Tensor(feature).cuda().unsqueeze(0)
#                 # print(user_timediference_list)
#                 user_timediffs_tensor = torch.Tensor([user_timediference_list[j]]).cuda().unsqueeze(0)
#                 action_timediffs_tensor = torch.Tensor([action_timediference_list[j]]).cuda().unsqueeze(0)
                
#                 action_embedding_previous = action_embeddings[previous_actionid,:].unsqueeze(0)

#                 actual_user_embedding_static = user_embedding_static[userid,:].unsqueeze(0)
#                 previous_action_embedding_static = action_embedding_static[previous_actionid,:].unsqueeze(0)

#                 # print(user_timediffs_tensor)
#                 # PROJECT USER EMBEDDING TO CURRENT TIME
#                 user_embedding_input = user_embeddings[userid,:].unsqueeze(0)
#                 user_projected_embedding = model.forward(user_embedding_input, action_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
#                 user_action_embedding = torch.cat([user_projected_embedding, action_embedding_previous, previous_action_embedding_static, actual_user_embedding_static], dim=1)

#                 # PREDICT NEXT ACTION EMBEDDING                            
#                 predicted_action_embedding = model.predict_action_embedding(user_action_embedding)
                
#                 # CALCULATE PREDICTION LOSS
#                 action_embedding_input = action_embeddings[actionid,:].unsqueeze(0)
#                 action_embedding_static_sq = action_embedding_static[actionid,:].unsqueeze(0)
#                 loss += MSELoss(predicted_action_embedding, torch.cat([action_embedding_input, action_embedding_static_sq], dim=1).detach())

#                 # UPDATE DYNAMIC EMBEDDINGS AFTER action
#                 user_embedding_output = model.forward(user_embedding_input, action_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
#                 action_embedding_output = model.forward(user_embedding_input, action_embedding_input, timediffs=action_timediffs_tensor, features=feature_tensor, select='action_update')
                

                
#                 # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
#                 loss += MSELoss(action_embedding_output, action_embedding_input.detach())
#                 loss += MSELoss(user_embedding_output, user_embedding_input.detach())

#                 # loss.requires_grad = True
#                 # BACKPROPAGATE ERROR 
#                 total_loss += loss.item()
#                 mean_loss.append(loss.item())
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()

#                 user_embeddings[userid,:] = user_embedding_output
#                 action_embeddings[actionid,:] = action_embedding_output

#                 user_embeddings_timeseries[j,:] = user_embedding_output
#                 action_embeddings_timeseries[j,:] = action_embedding_output

#                 loss = 0
#                 action_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
#                 user_embeddings.detach_()
#                 action_embeddings_timeseries.detach_() 
#                 user_embeddings_timeseries.detach_()

#                 if not j%100:
#                     wandb.log({'epoch': ep,
#                                'total_loss_mean': np.mean(mean_loss)})
        
#         print("\nLast epoch took {} minutes".format((time.time()-epoch_start_time)/60))
#         print("\nTotal loss in this epoch = %f\n" % (total_loss))
        
#         ##### EVALUATION PHASE #####
#         model.eval()
        
#         val_loss = []

#         with trange(validation_start_idx, test_end_idx) as progress_bar_evaluation:
#             for j in progress_bar_evaluation:
#                 progress_bar_evaluation.set_description('Processed EVALUATION %dth interactions' % j)

#                 # GET DATA FROM LISTS
#                 userid = user_list_id[j]
#                 actionid = action_list_id[j]
#                 feature = feature_list[j]
#                 previous_actionid = user_previous_actionid_list[j]


#                 # CREATE AND FORMAT TENSORS
#                 user_embedding_input = user_embeddings[userid,:].unsqueeze(0)
#                 action_embedding_previous = action_embeddings[user_previous_actionid_list[j],:].unsqueeze(0)
#                 feature_tensor = torch.Tensor(feature).cuda().unsqueeze(0)

#                 user_timediffs_tensor = torch.Tensor([user_timediference_list[j]]).cuda().unsqueeze(0)
#                 action_timediffs_tensor = torch.Tensor([action_timediference_list[j]]).cuda().unsqueeze(0)
                

#                 actual_user_embedding_static = user_embedding_static[userid,:].unsqueeze(0)
#                 previous_action_embedding_static = action_embedding_static[previous_actionid,:].unsqueeze(0)

#                 # PROJECT USER EMBEDDING TO CURRENT TIME
#                 user_projected_embedding = model.forward(user_embedding_input, action_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
#                 user_action_embedding = torch.cat([user_projected_embedding, action_embedding_previous, previous_action_embedding_static, actual_user_embedding_static], dim=1)

#                 # PREDICT NEXT ACTION EMBEDDING                            
#                 predicted_action_embedding = model.predict_action_embedding(user_action_embedding)
                
#                 # CALCULATE PREDICTION LOSS
#                 action_embedding_input = action_embeddings[actionid,:].unsqueeze(0)
#                 action_embedding_static_sq = action_embedding_static[actionid,:].unsqueeze(0)
#                 loss += MSELoss(predicted_action_embedding, torch.cat([action_embedding_input, action_embedding_static_sq], dim=1).detach())

#                 # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS 
#                 euclidean_distances = nn.PairwiseDistance()(predicted_action_embedding.repeat(num_actions, 1), torch.cat([action_embeddings, action_embedding_static], dim=1)).squeeze(-1) 
                
#                 # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
#                 true_action_distance = euclidean_distances[actionid]
#                 euclidean_distances_smaller = (euclidean_distances < true_action_distance).data.cpu().numpy()
#                 true_action_rank = np.sum(euclidean_distances_smaller) + 1
#                 # print(euclidean_distances)
#                 # print(true_action_distance)
#                 # print(euclidean_distances_smaller)
                

#                 validation_ranks.append(true_action_rank)

#                 # UPDATE USER AND ITEM EMBEDDING
#                 user_embedding_output = model.forward(user_embedding_input, action_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update') 
#                 action_embedding_output = model.forward(user_embedding_input, action_embedding_input, timediffs=action_timediffs_tensor, features=feature_tensor, select='action_update') 

#                 # SAVE EMBEDDINGS
#                 user_embeddings[userid,:] = user_embedding_output.squeeze(0) 
#                 action_embeddings[actionid,:] = action_embedding_output.squeeze(0) 
#                 user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
#                 action_embeddings_timeseries[j, :] = action_embedding_output.squeeze(0)

#                 # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
#                 loss += MSELoss(user_embedding_output, user_embedding_input.detach())
#                 loss += MSELoss(action_embedding_output, action_embedding_input.detach())

#                 val_loss.append(loss.item())



        
        
#         ##### TEST PHASE #####
#         # if not ep%5:
#         #     with trange(test_start_idx, test_end_idx) as progress_bar4:
#         #         for j in progress_bar4:
#         #             print('')


#         ##### PERFORMANCE METRICS #####

#         performance_dict = dict()

#         ranks = validation_ranks
#         mrr = np.mean([1.0 / r for r in ranks])
#         rec = sum(np.array(ranks) <= 1)*1.0 / len(ranks)
#         performance_dict['validation'] = [mrr, rec]

#         # PRINT AND SAVE THE PERFORMANCE METRICS
#         wandb.log({'epoch': ep,
#                    'mrr': mrr,
#                    'rec': rec,
#                    'loss_val_mean': np.mean(val_loss)})
#         fw = open(output_fname, "a")
#         metrics = ['Mean Reciprocal Rank', 'Recall']
#         print('\n\n*** Validation performance of epoch %d ***' % ep)
#         fw.write('\n\n*** Validation performance of epoch %d ***\n' % ep)
#         for i in range(len(metrics)):
#             print(metrics[i] + ': ' + str(performance_dict['validation'][i]))
#             fw.write("Validation: " + metrics[i] + ': ' + str(performance_dict['validation'][i]) + "\n")
        
#         fw.flush()
#         fw.close()      
#         # END OF ONE EPOCH 

#         user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
#         action_embeddings_dystat = torch.cat([action_embeddings, action_embedding_static], dim=1)

#         if not ep%5:
#             save_model(model, optimizer, args, ep, user_embeddings_dystat, action_embeddings_dystat, train_end_idx, user_embeddings_timeseries, action_embeddings_timeseries)
            

#         user_embeddings = initial_user_embedding.repeat(num_users, 1)
#         action_embeddings = initial_action_embedding.repeat(num_actions, 1)

# # END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
# print("\n\n*** Training complete. Saving final model. ***\n\n")
# save_model(model, optimizer, args, ep, user_embeddings_dystat, action_embeddings_dystat, train_end_idx, user_embeddings_timeseries, action_embeddings_timeseries)

