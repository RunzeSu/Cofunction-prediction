import numpy as np
import pandas as pd
import torch

def preprocess():
              
    unique_genes = []
    with open('/mnt/ufs18/home-052/surunze/gene_interaction/Pathways-of-gene-All--in-Saccharomyces-cerevisiae-S288c_metacyc.txt') as f:
        record = f.readlines()
    
    genes = []
    for i, item in enumerate(record):
        ls = item.split("\t")[1].split(" // ")
        ls[-1] = ls[-1][:-1]
        if len(ls) > 1:
            genes.append(ls)
            unique_genes += ls
    
    ft_data = np.array(pd.read_csv("/mnt/ufs18/home-052/surunze/gene_interaction/all_pathways_with_fitness_metacyc.csv"))
    for item in ft_data:
        unique_genes += item[0].split("_")
    
    unique_genes = list(set(unique_genes))
    unique_genes.sort()
    unique_genes = {unique_genes[i]: i for i in range(len(unique_genes))}
    adj_matrix = np.zeros((len(unique_genes), len(unique_genes)))
    
    for k, connection in enumerate(genes):
        for i in range(0, len(connection) - 1):
            for j in range(i, len(connection)):
                gene_1 = connection[i]
                gene_2 = connection[j]
                index_1 = unique_genes[gene_1]
                index_2 = unique_genes[gene_2]
                if adj_matrix[index_1][index_2] == 0:
                    adj_matrix[index_1][index_2] = 1
                    adj_matrix[index_2][index_1] = 1
    
    y = ft_data[:, 1]
    node_embeddings = np.identity(len(unique_genes))
    node_value = np.zeros((len(ft_data), 3))
        
    edges = []
    for connection in genes:
        for i in range(0, len(connection) - 1):
            for j in range(i+1, len(connection)):
                gene_index_1 = unique_genes[connection[i]]
                gene_index_2 = unique_genes[connection[j]]
                edges.append([gene_index_1, gene_index_2])
    
    edges = set(map(tuple, np.unique(np.array(edges), axis = 0)))

    index = np.zeros([len(ft_data), 2])
    for i, name in enumerate(ft_data[:, 0]):
        gene_1, gene_2 = name.split("_")
        index[i, 0] = unique_genes[gene_1]
        index[i, 1] =  unique_genes[gene_2]
        node_value[i, 0] = ft_data[i, 2]
        node_value[i, 1] = ft_data[i, 3]
        node_value[i, 2] = ft_data[i, 4]
    
    training_sample = 140080
    testing_sample = 150220
    training_value = node_value[0:training_sample]
    dev_value = node_value[training_sample:testing_sample]
    testing_value = node_value[testing_sample:]
    training_index = index[0:training_sample]
    dev_index = index[training_sample:testing_sample]
    testing_index = index[testing_sample:]
    training_y = y[0:training_sample]
    dev_y = y[training_sample:testing_sample]
    testing_y = y[testing_sample:]
    
    for k, [i, j] in enumerate(dev_index):
        if dev_y[k] == 1:
            if (int(i), int(j)) in edges:
                edges.remove((int(i), int(j)))
            if (int(j), int(i)) in edges:
                edges.remove((int(j), int(i)))
    
    for k, [i, j] in enumerate(testing_index):
        if testing_y[k] == 1:
            if (int(i), int(j)) in edges:
                edges.remove((int(i), int(j)))
            if (int(j), int(i)) in edges:
                edges.remove((int(j), int(i)))
        
    return torch.tensor(node_embeddings), torch.tensor(adj_matrix),  edges, [torch.tensor(training_value), torch.tensor(dev_value), torch.tensor(testing_value)], [torch.tensor(training_index), torch.tensor(dev_index), torch.tensor(testing_index)], [torch.tensor(training_y.astype(float)), torch.tensor(dev_y.astype(float)), torch.tensor(testing_y.astype(float))]



























