import os
import torch
import gpustat
import numpy as np
from collections import Counter
from data_loader import *

PATH = './'

# SELECT THE GPU WITH MOST FREE MEMORY TO SCHEDULE JOB 
def select_free_gpu():
    mem = []
    gpus = list(set(range(torch.cuda.device_count()))) # list(set(X)) is done to shuffle the array
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
    return str(gpus[np.argmin(mem)])

def prepare_data_folder(args, path=PATH):
    directory = os.path.join(path, 'embeddings/%s' % args.dataname)
    if not os.path.exists(directory):
        os.makedirs(directory)


def draw_initial_embeddings(model, name, user_list_id, feature_list, user_timediference_list, user_previous_actionid_list, action_embeddings, user_embeddings, head_labels):
    users_first_tensor = None
    users_first_labels = []
    for user in set(user_list_id):
        userid = user
        first_interaction = user_list_id.index(userid)
        feature = feature_list[first_interaction]
        feature_tensor = torch.Tensor(feature).cuda().unsqueeze(0)

        user_timediffs_tensor = torch.Tensor([user_timediference_list[first_interaction]]).cuda().unsqueeze(0)
        previous_actionid = user_previous_actionid_list[first_interaction]
        action_embedding_previous = action_embeddings[previous_actionid,:].unsqueeze(0)

        user_embedding_input = user_embeddings[userid,:].unsqueeze(0)
        user_projected_embedding = model.forward(user_embedding_input, action_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
        
        if users_first_tensor is None:
            users_first_tensor = user_projected_embedding
        else:
            users_first_tensor = torch.cat((users_first_tensor, user_projected_embedding), dim=0)

        users_first_labels.append('First')
    d = {'user':list(set(user_list_id)), 'label':users_first_labels}
    labels_df = pd.DataFrame(d)
    labels_df.to_csv('labels_'+name+'.tsv', sep='\t', header=False, index=False)
    x_np = users_first_tensor.cpu().detach().numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv('embeddings_'+name+'.tsv', sep='\t', header=False, index=False)



def draw_final_embeddings(name, user_embeddings_dystat, user_list_id):
    users_final_tensor = None
    users_final_labels = []
    for embedding in user_embeddings_dystat:
        embedding = embedding[0:128]
        if users_final_tensor is None:
            users_final_tensor = embedding
            print(users_final_tensor.shape)
            users_final_tensor = users_final_tensor.unsqueeze(0)
        else:
            users_final_tensor = torch.cat((users_final_tensor, embedding.unsqueeze(0)), dim=0)
        users_final_labels.append('End')
    d = {'user':list(set(user_list_id)), 'label':users_final_labels}
    labels_df = pd.DataFrame(d)
    labels_df.to_csv('labels_'+name+'.tsv', sep='\t', header=False, index=False, mode='a')
    x_np = users_final_tensor.cpu().detach().numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv('embeddings_'+name+'.tsv', sep='\t', header=False, index=False, mode='a')
    
def draw_embeddings(args, f, state, path=PATH):
    embedding_path = os.path.join(path, args.folder, f)
    embeddings = torch.load(embedding_path)        
    embedding_tensor = None
    labels = []
    for embedding in embeddings:
        embedding = embedding[0:args.embedding_size]
        if embedding_tensor is None:
            embedding_tensor = embedding.unsqueeze(0)
        else:
            embedding_tensor = torch.cat((embedding_tensor, embedding.unsqueeze(0)), dim=0)
        labels.append(state)
    
    d = {'label':labels}
    
    # WRITE LABELS TO FILE
    filename = os.path.join(path, 'embeddings/%s/labels_%s.tsv' % (args.dataname, args.dataname))
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write('label\n')
    labels_df = pd.DataFrame(d)
    labels_df.to_csv(filename, sep='\t', header=False, index=False, mode='a')
    
    # WRITE EMBEDDINGS TO FILE
    filename = os.path.join(path, 'embeddings/%s/embeddings_%s.tsv' % (args.dataname, args.dataname))
    x_np = embedding_tensor.cpu().detach().numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv(filename, sep='\t', header=False, index=False, mode='a')
    