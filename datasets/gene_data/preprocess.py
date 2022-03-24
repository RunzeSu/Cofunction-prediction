import numpy as np
import pandas as pd

def preprocess():

    unique_genes = []
    with open('/mnt/ufs18/home-052/surunze/gene_interaction/Pathways-of-gene-All--in-Saccharomyces-cerevisiae-S288c_metacyc.txt') as f:
        record = f.readlines()
    
    genes = []
    for i, item in enumerate(record):
        ls = item.split("\t")[1].split(" // ")
        ls[-1] = ls[-1][:-2]
        if len(ls) > 1:
            genes.append(ls)
            unique_genes += ls
    
    ft_data = np.array(pd.read_csv("/mnt/ufs18/home-052/surunze/gene_interaction/all_pathways_with_fitness_metacyc.csv"))
    for item in ft_data:
        unique_genes += item[0].split("_")
    
    """
    adj_matrix = np.zeros((len(unique_genes), len(unique_genes)))
    for connection in genes:
    for i in range(0, len(connection) - 1):
        for j in range(i, len(connection)):
            gene_1 = connection[i]
            gene_2 = connection[j]
            index_1 = gene_index[gene_1]
            index_2 = gene_index[gene_2]
            if adj_matrix[index_1][index_2] == 0:
                adj_matrix[index_1][index_2] = 1
                adj_matrix[index_2][index_1] = 1
    """
    
    unique_genes = list(set(unique_genes))
    unique_genes = {unique_genes[i]: i for i in range(len(unique_genes))}
    y = ft_data[:, 1]
    node_embeddings = np.identity(len(unique_genes))
    node_value = np.zeros((len(ft_data), 5))
    
    edges = []
    for connection in genes:
        for i in range(0, len(connection) - 1):
            for j in range(i+1, len(connection)):
                gene_index_1 = unique_genes[connection[i]]
                gene_index_2 = unique_genes[connection[j]]
                edges.append([gene_index_1, gene_index_2])
    
    edges = np.unique(np.array(edges), axis = 0)
    
    for i, name in enumerate(ft_data[:, 0]):
        gene_1, gene_2 = name.split("_")
        node_value[i, 0] = unique_genes[gene_1]
        node_value[i, 1] = unique_genes[gene_2]
        node_value[i, 2] = ft_data[i, 2]
        node_value[i, 3] = ft_data[i, 3]
        node_value[i, 4] = ft_data[i, 4]
        
    training_sample = 140080
    testing_sample = 150220
    training_value = node_value[0:training_sample]
    dev_value = node_value[training_sample:testing_sample]
    testing_value = node_value[testing_sample:]
    training_y = y[0:training_sample]
    dev_y = y[training_sample:testing_sample]
    testing_y = y[testing_sample:]
    
    return graph, edges, [training_value, dev_value, testing_value], [training_y, dev_y, testing_y]




















