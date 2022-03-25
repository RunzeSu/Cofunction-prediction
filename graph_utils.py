import dgl
import dgl.nn as dglnn
from scipy.sparse import csc_matrix
import numpy as np
import torch
import torch.nn.functional as F

def gen_bf_features(bf_info):
    pass

def feature_graph_base(feature_mod1, feature_mod1_test=None, bf_info=None, device='cuda'):

    input_train_mod1 = csc_matrix(feature_mod1)

    if feature_mod1_test is not None:
        input_test_mod1 = csc_matrix(feature_mod1_test)
        assert(input_test_mod1.shape[1] == input_train_mod1.shape[1])
        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0]+i) for i, t in enumerate(input_train_mod1)] +\
                   [np.array(t.nonzero()[0]+i+input_train_mod1.shape[0]) for i, t in enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] +\
                   [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        sample_size = input_train_mod1.shape[0]+input_test_mod1.shape[0] # total nnumber of cells: 81241, training, valid, test
        weights = torch.from_numpy(np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float() 

    else:
        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0]+i) for i, t in enumerate(input_train_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))
        sample_size = input_train_mod1.shape[0]
        weights = torch.from_numpy(np.concatenate([input_train_mod1.tocsr().data], axis=0)).float()

    graph_data = {
        ('cell', 'cell2feature', 'feature'): (u, v),
        ('feature', 'feature2cell', 'cell'): (v, u),
    }

    """
    Graph(num_nodes={'cell': 81241, 'feature': 390},
      num_edges={('cell', 'cell2feature', 'feature'): 31683989, ('feature', 'feature2cell', 'cell'): 31683989},
      metagraph=[('cell', 'feature', 'cell2feature'), ('feature', 'cell', 'feature2cell')])
    """
    g = dgl.heterograph(graph_data)

    if bf_info:
        g.nodes['cell'].data['bf'] = gen_bf_features(bf_info)
    else:
        g.nodes['cell'].data['bf'] = torch.zeros(sample_size).float()

    g.nodes['cell'].data['id'] = torch.arange(sample_size).long()
#     g.nodes['cell'].data['source'] = 
    g.nodes['feature'].data['id'] = torch.arange(input_train_mod1.shape[1]).long()
    g.edges['cell2feature'].data['weight'] = g.edges['feature2cell'].data['weight'] = weights
    
    g = g.to(device)
    return g

def feature_graph_propagation(g, layers=3, alpha=0.5, beta=0.5, cell_init=None, feature_init='id', device='cuda', verbose=True):
    
#     assert(layers>2)
    
    # gconv = dglnn.HeteroGraphConv({
    #                 'cell2feature' : dglnn.GraphConv(in_feats=0, out_feats=0, norm='none', weight=False, bias=False),
    #                 'feature2cell' : dglnn.GraphConv(in_feats=0, out_feats=0, norm='none', weight=False, bias=False),
    #             },
    #         aggregate='sum')
    

    gconv = dglnn.HeteroGraphConv({
                    'cell2feature' : dglnn.GraphConv(in_feats=0, out_feats=0, norm='right', weight=False, bias=False),
                    'feature2cell' : dglnn.GraphConv(in_feats=0, out_feats=0, norm='right', weight=False, bias=False),
                },
            aggregate='sum')
    
    if feature_init is None:
        feature_X = torch.zeros((g.nodes('feature').shape[0], g.srcdata[cell_init]['cell'].shape[1])).to(device)
    elif feature_init == 'id':
        # feature_X.shape: torch.Size([390, 390]), each feature node intialized with one hot embed
        feature_X = F.one_hot(g.srcdata['id']['feature']).float().to(device) 
    
    
    if cell_init is None:
        # cell_X.shape: ([81241, 390]), each cell node initialized with zero embed
        cell_X = torch.zeros(g.nodes('cell').shape[0], feature_X.shape[1]).to(device)
    else:
        cell_X = g.srcdata[cell_init]['cell']
    
    h = {'feature': feature_X,
         'cell': cell_X}
    hcell = []
    for i in range(layers):
        # h1: {'cell': ([81241, 390]), 'feature': ([390, 390])}
        h1 = gconv(g, h, mod_kwargs={'cell2feature':{'edge_weight': g.edges['cell2feature'].data['weight']},
                                     'feature2cell':{'edge_weight': g.edges['feature2cell'].data['weight']}})
        if verbose: print(i, 'cell', h['cell'].abs().mean(), h1['cell'].abs().mean())
        if verbose: print(i, 'feature', h['feature'].abs().mean(), h1['feature'].abs().mean())

        # to comment out
        h1['feature'] = (h1['feature'] - h1['feature'].mean())/(h1['feature'].std() if h1['feature'].sum()!=0 else 1) ## TUNE (remove)
        h1['cell'] = (h1['cell'] - h1['cell'].mean())/(h1['cell'].std() if h1['cell'].sum()!=0 else 1)

        # import ipdb
        # ipdb.set_trace()

        h = {'feature': h['feature']*alpha + h1['feature']*(1-alpha), 
             'cell': h['cell']*beta + h1['cell']*(1-beta)}

        # to comment out
        h['feature'] = (h['feature'] - h['feature'].mean())/h['feature'].std()  # TUNE (remove?)
        h['cell'] = (h['cell'] - h['cell'].mean())/h['cell'].std()

        hcell.append(h['cell'])
        # hcell.append(h['cell']*2)
        
    if verbose: print(hcell[-1].abs().mean())

    return hcell[1:]








