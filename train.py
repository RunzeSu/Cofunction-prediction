import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from autoencoder_model import Graph_encoder
from config.config import parse_args
from preprocess import preprocess

os.environ["CUDA_VISIBLE_DEVICES"]="1"

torch.manual_seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

    ## Required parameters
parser.add_argument("--dataset",
                    default=None,
                    type=str,
                    help="The input data dir.")
parser.add_argument("--enc_in_channels",
                    default=1109,
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
                    default=400,
                    type=int,
                    help="Number of epochs")
parser.add_argument("--batch_size",
                    default=8,
                    type=int,
                    help="Learning Rate")
parser.add_argument("--lr",
                    default=0.0001,
                    type=float,
                    help="Learning Rate")

args = parser.parse_args()


model = Graph_encoder(args).to(device)
optimizer = Adam(model.parameters(), lr=0.0001)
graph_embedding, adj_matrix, edges, [training_value, dev_value, testing_value], [training_index, dev_index, testing_index], [training_y, dev_y, testing_y] = preprocess()
graph_embedding, adj_matrix, edges, training_value, dev_value, testing_value, training_index, dev_index, testing_index, training_y, dev_y, testing_y = graph_embedding, adj_matrix, edges, training_value, dev_value, testing_value, training_index, dev_index, testing_index, training_y, dev_y, testing_y


n_epochs = args.epoch # or whatever
batch_size = args.batch_size # or whatever
length = training_value.size()[0]

for epoch in range(n_epochs):
    print("Epoch number ", epoch)
    model.train()
    optimizer.zero_grad()
    loss = torch.nn.BCELoss()
    permutation = torch.randperm(length)
    #graph_embedding = graph_embedding.repeat(batch_size, 1, 1)
    #edges = edges.repeat(batch_size, 1, 1)
    print("Training start")
    for i in range(0,length, batch_size):
        
        indices = permutation[i:i+batch_size]
        batch_value = training_value[indices]
        batch_value.to(device)
        batch_index = training_index[indices]
        batch_index = batch_index.long().to(device)
        batch_y = training_y[indices]
        batch_y = batch_y.to(device)
        pred = model(graph_embedding.to(device), edges.to(device), batch_value, batch_index)
        loss = loss(pred,batch_y)
        loss.backward()
        optimizer.step()
"""  
    if epoch % 2 == 0:
        model.eval()
        roc_auc, ap = model.single_test(graph_embedding, adj_matrix, testing_value)
"""     