import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale



def load_adj(features, normalization=True, normalization_type='normalize',
              k_nearest_neighobrs=10, prunning_one=False, prunning_two=True , common_neighbors=2):
    if normalization:
        if normalization_type == 'minmax_scale':
            features = minmax_scale(features)
        elif normalization_type == 'maxabs_scale':
            features = maxabs_scale(features)
        elif normalization_type == 'normalize':
            features = normalize(features)
        elif normalization_type == 'robust_scale':
            features = robust_scale(features)
        elif normalization_type == 'scale':
            features = scale(features)
        elif normalization_type == '255':
            features = np.divide(features, 255.)
        elif normalization_type == '50':
            features = np.divide(features, 50.)
        else:
            print("Please enter a correct normalization type!")

    # construct three kinds of adjacency matrix

    adj, adj_wave, adj_hat = construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one,
                                                        prunning_two, common_neighbors)
    return adj, adj_hat


def construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors):
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs+1, algorithm='ball_tree').fit(features)
    adj_wave = nbrs.kneighbors_graph(features)  # <class 'scipy.sparse.csr.csr_matrix'>

    if prunning_one:
        # Pruning strategy 1
        original_adj_wave = adj_wave.A
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)
    else:
        # transform the matrix to be symmetric (Instead of Pruning strategy 1)
        np_adj_wave = construct_symmetric_matrix(adj_wave.A)
        adj_wave = sp.csc_matrix(np_adj_wave)

    # obtain the adjacency matrix without self-connection
    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.A
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()

    # construct the adjacency hat matrix
    adj_hat = construct_adjacency_hat(adj)  

    return adj, adj_wave, adj_hat


def construct_adjacency_hat(adj):
    """
    :param adj: original adjacency matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def construct_symmetric_matrix(original_matrix):
    """
        transform a matrix (n*n) to be symmetric
    :param np_matrix: <class 'numpy.ndarray'>
    :return: result_matrix: <class 'numpy.ndarray'>
    """
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")

    return result_matrix


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  
    values = sparse_mx.data  
    shape = sparse_mx.shape  
    return coords, values, shape

def load_adjacency_multiview(multi_view_features, k):
    adj_list = []
    adj_hat_list = []

    for idx, features in enumerate(multi_view_features[0]):
        adj, adj_hat = load_adj(features, k_nearest_neighobrs = k)
        adj_list.append(adj.todense())
        adj_hat_list.append(adj_hat.todense())

    adj_list = np.array(adj_list)
    adj_hat_list = np.array(adj_hat_list)

    
    return adj_list, adj_hat_list
