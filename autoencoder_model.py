import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        return mu

class Graph_encoder(nn.Module):
    def __init__(self, args):
        super(Graph_encoder, self).__init__()
        self.encoder = Encoder(args.enc_in_channels,
                                args.enc_hidden_channels,
                                args.enc_out_channels)
        self.decoder = torch.nn.Linear(args.enc_out_channels*2+3, args.hidden_dimension)
        self.pred_layer = torch.nn.Linear(args.hidden_dimension, 1)
        

    def forward(self, x, edge_index, value, index):
        
        all_pred = []
        for k in range(len(edge_index)):
        
            index1 = index[k:k+1, 0]
            index2 = index[k:k+1, 1]
            hidden1 = self.encoder(x, edge_index[k])[index1]
            hidden2 = self.encoder(x, edge_index[k])[index2]
            hidden = torch.cat((hidden1, hidden2, value[k:k+1]), 1)
            output = self.decoder(hidden)
            output = torch.nn.ReLU()(output)
            pred = self.pred_layer(output)
            pred = torch.nn.Sigmoid()(pred)
            all_pred.append(torch.flatten(pred))
        
        return torch.cat(all_pred)

    def run_eval(self, x, edge_index, value, index, y):
        with torch.no_grad():
            index1 = index[:, 0]
            index2 = index[:, 1]
            hidden1 = self.encoder(x, edge_index)
            hidden1 = self.encoder(x, edge_index)[index1]
            hidden2 = self.encoder(x, edge_index)[index2]
            hidden = torch.cat((hidden1, hidden2, value), 1)
            output = self.decoder(hidden)
            output = torch.nn.ReLU()(output)
            pred = self.pred_layer(output)
            pred = torch.flatten(torch.nn.Sigmoid()(pred))
        pred = pred.to('cpu').detach().numpy()
        auc_roc = roc_auc_score(y, pred)
        precision, recall, thresholds = precision_recall_curve(y, pred)
        auc_precision_recall = auc(recall, precision)
        
        return pred, auc_roc, auc_precision_recall


"""
class DeepVGAE(VGAE):
    def __init__(self, args):
        super(DeepVGAE, self).__init__(encoder=Encoder(args.enc_in_channels,
                                                          args.enc_hidden_channels,
                                                          args.enc_out_channels),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, all_edge_index):
        z = self.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.kl_loss()

        return pos_loss + neg_loss + kl_loss

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score
"""