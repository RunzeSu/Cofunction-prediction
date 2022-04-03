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
parser.add_argument("--model_path",
                    default="/mnt/ufs18/home-052/surunze/gene_interaction/VGAE_pyG/saved_models/model_epoch_",
                    type=str,
                    help="Path for trained model")

args = parser.parse_args()


model = Graph_encoder(args).to(device)
optimizer = Adam(model.parameters(), lr=0.0001)
graph_embedding, adj_matrix, edges, [training_value, dev_value, testing_value], [training_index, dev_index, testing_index], [training_y, dev_y, testing_y] = preprocess()
loss_func = torch.nn.BCELoss()
length = training_value.size()[0]

for k in range(1, 30):
    model.load_state_dict(torch.load(args.model_path+str(k)))
    model.eval()
    
    
    pred, auc_roc, auc_precision_recall = model.run_eval(graph_embedding.float().to(device), torch.tensor(list(edges)).T.to(device), testing_value.float().to(device), testing_index.long().to(device), testing_y.float().to('cpu').numpy())  
    eval_loss = loss_func(torch.tensor(pred).to(device), testing_y.float().to(device))
    eval_acc = torch.tensor(pred), testing_y.float()
    
    print("Testing Loss = ", eval_loss.to("cpu").detach().numpy())
    print("Testing Accuracy = ", float(sum((pred >= 0.5) == (testing_y.int().numpy() == 1))/len(pred)))
    print("ROC AUC in Testing set = ", auc_roc)
    print("PR AUC in Testing set = ", auc_precision_recall)