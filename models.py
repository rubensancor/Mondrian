import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from collections import defaultdict
import numpy as np
import math, random, sys

PATH = "./"

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

class Mondrian(nn.Module):
    def __init__(self, args, num_users, num_actions, num_features):
        super(Mondrian, self).__init__()

        print('*** Initializing MONDRIAN model ***')
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_actions = num_actions
        self.user_static_embedding_size = num_users
        self.action_static_embedding_size = num_actions

        print("Initializing user and interaction embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(self.embedding_dim))
        self.initial_action_embedding = nn.Parameter(torch.Tensor(self.embedding_dim))
        
        # TODO: Decidir que features meterle aqui para ponerle el +x por features
        # +1 for the number of features, in this case is the interaction type but we can add centralities to them
        rnn_input_size_users = rnn_input_size_actions = self.embedding_dim + 1 + num_features

        print("Initializing user and interaction RNNs")
        self.user_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.action_rnn = nn.RNNCell(rnn_input_size_actions, self.embedding_dim)

        print("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        # print('SIZE: %i' % (self.user_static_embedding_size + self.interaction_static_embedding_size + self.embedding_dim * 2))
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.action_static_embedding_size + self.embedding_dim * 2, self.action_static_embedding_size + self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        print("*** MONDRIAN initialization complete ***\n\n")

    def forward(self, user_embeddings, action_embeddings, timediffs=None, features=None, select=None):
        if select == 'action_update':
            input1 = torch.cat([user_embeddings, timediffs, features], dim=1)
            action_embedding_output = self.action_rnn(input1, action_embeddings)
            return F.normalize(action_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([action_embeddings, timediffs, features], dim=1)
            user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)
        
        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            return user_projected_embedding
    
    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)

        return X_out

    def predict_action_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out

def to_one_hot(tensor):
    index = torch.argmax(tensor, dim=0).cuda()
    one_hot = torch.FloatTensor(tensor.shape).cuda()
    one_hot.zero_()
    one_hot = one_hot.scatter(0, index, 1)
    return one_hot




# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_action, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_action
    global tbatchid_user, tbatchid_action, current_tbatches_user_timediffs, current_tbatches_action_timediffs, current_tbatches_user_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_action = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_action = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_action_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a action is in
    tbatchid_action = defaultdict(lambda: -1)

    global total_reinitialization_count

# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, args, epoch, user_embeddings, action_embeddings, train_end_idx, user_embeddings_time_series=None, action_embeddings_time_series=None, path=PATH):
    print("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'action_embeddings': action_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx
            }

    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['action_embeddings_time_series'] = action_embeddings_time_series.data.cpu().numpy()

    directory = os.path.join(path, 'saved_models/%s' % args.data)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, "checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.model, epoch, args.train_proportion))
    torch.save(state, filename, pickle_protocol=4)
    print("*** Saved embeddings and model to file: %s ***\n\n" % filename)

# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, epoch):
    modelname = args.model
    filename = PATH + "saved_models/%s/checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.data, modelname, epoch, args.train_proportion)
    checkpoint = torch.load(filename)
    print("Loading saved embeddings and model: %s" % filename)
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    action_embeddings = Variable(torch.from_numpy(checkpoint['action_embeddings']).cuda())
    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        action_embeddings_time_series = Variable(torch.from_numpy(checkpoint['action_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        action_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return (model, optimizer, user_embeddings, action_embeddings, user_embeddings_time_series, action_embeddings_time_series, train_end_idx)

# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD 
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()
