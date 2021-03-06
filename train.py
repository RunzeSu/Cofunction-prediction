import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


from autoencoder_model import Graph_encoder
from preprocess import preprocess

os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.manual_seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

    ## Required parameters
parser.add_argument("--dataset",
                    default=None,
                    type=str,
                    help="The input data dir.")
parser.add_argument("--enc_in_channels",
                    default=972,
                    type=int,
                    help="The input dimension")
parser.add_argument("--enc_hidden_channels",
                    default=16,
                    type=int,
                    help="The hidden dimension")
parser.add_argument("--enc_out_channels",
                    default=100,
                    type=int,
                    help="The bottleneck dimension of the encoder")
parser.add_argument("--hidden_dimension",
                    default=10,
                    type=int,
                    help="The hidden dimension of the decoder")
parser.add_argument("--epoch",
                    default=30,
                    type=int,
                    help="Number of epochs")
parser.add_argument("--batch_size",
                    default=128,
                    type=int,
                    help="Learning Rate")
parser.add_argument("--lr",
                    default=0.0001,
                    type=float,
                    help="Learning Rate")
parser.add_argument("--model_path",
                    default="/mnt/ufs18/home-052/surunze/gene_interaction/VGAE_pyG/saved_models",
                    type=str,
                    help="Path for trained model")

args = parser.parse_args()


model = Graph_encoder(args).to(device)
optimizer = Adam(model.parameters(), lr=0.0001)
graph_embedding, adj_matrix, edges, [training_value, dev_value, testing_value], [training_index, dev_index, testing_index], [training_y, dev_y, testing_y] = preprocess()

n_epochs = args.epoch 
batch_size = args.batch_size 
length = training_value.size()[0]

"""
batch_edges = []
batch_graph = []
batch_value = []
batch_index = []
batch_y = []
for k in range(len(training_value)):
    edges_copy = edges.copy()
    if training_y[k] == 1:
        i, j = training_index[k, 0], training_index[k, 1]
        if (int(i), int(j)) in edges_copy:
            edges_copy.remove((int(i), int(j)))
        if (int(j), int(i)) in edges_copy:
            edges_copy.remove((int(j), int(i)))
    batch_edges.append(torch.tensor(list(edges_copy)).T.to(device))
    batch_graph.append(graph_embedding.float().to(device))
    batch_value.append(training_value[k].float().to(device))
    batch_index.append(training_index[k].long().to(device))
    batch_y.append(training_y[k].float().to(device))
    training_data = Data(batch_edges = batch_edges, batch_graph = batch_graph, batch_value = batch_value, batch_index = batch_index, batch_y = batch_y)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
"""
epoch = 0
for epoch in range(n_epochs):
    print("Epoch number ", epoch)
    model.train()
    optimizer.zero_grad()
    loss_func = torch.nn.BCELoss()
    permutation = torch.randperm(length)
    for l in range(0,length, batch_size):
        indices = permutation[l:l+batch_size]
        batch_edges = []
        batch_graph = []
        for k in indices:
            edges_copy = edges.copy()
            if training_y[k] == 1:
                i, j = training_index[k, 0], training_index[k, 1]
                if (int(i), int(j)) in edges_copy:
                    edges_copy.remove((int(i), int(j)))
                    edges_copy.add((0, 0))
                if (int(j), int(i)) in edges_copy:
                    edges_copy.remove((int(j), int(i)))
                    edges_copy.add((1, 1))
            edges_copy = map(list, edges_copy)
            batch_edges.append(torch.tensor(list(edges_copy)).T)
            
        batch_edges = torch.stack(batch_edges, dim=0)
        batch_graph = graph_embedding.float()
        batch_value = training_value[indices].float().to(device)
        batch_index = training_index[indices].long().to(device)
        batch_y = training_y[indices].float().to(device)
        pred = model(batch_graph.to(device), batch_edges.to(device), batch_value, batch_index)
        loss = loss_func(pred,batch_y)
        loss.backward()
        optimizer.step()
    print("Training Loss = ", loss.to("cpu").detach().numpy())
    if epoch % 1 == 0:
        model.eval()
        pred, auc_roc, auc_precision_recall = model.run_eval(graph_embedding.float().to(device), torch.tensor(list(edges)).T.to(device), dev_value.float().to(device), dev_index.long().to(device), dev_y.float().to('cpu').numpy())  
        eval_loss = loss_func(torch.tensor(pred).to(device), dev_y.float().to(device))
        eval_acc = torch.tensor(pred), dev_y.float()
        
        print("Training Loss = ", loss)
        print("Validation Loss = ", eval_loss.to("cpu").detach().numpy())
        print("Validation Accuracy = ", float(sum((pred >= 0.5) == (dev_y.int().numpy() == 1))/len(pred)))
        print("ROC AUC in validation set = ", auc_roc)
        print("PR AUC in validation set = ", auc_precision_recall)
        torch.save(model.state_dict(), args.model_path+"/model_epoch_"+str(epoch))