import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.LGCN import LGCN
from model.FC_net import FCnet
import numpy as np
from utils.dataLoader import dataloader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from model.Autoencoder import Autoencoder
import random

def norm_2(x, y):
    return 0.5 * (torch.norm(x-y) ** 2)

def test(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    features, gnd, p_labeled, p_unlabeled, adjs, adj_hats = dataloader(args.dataset_name, args.k)

    num_class = np.unique(gnd).shape[0] 
    view_num = len(adjs)

    N = gnd.shape[0] 

    gnd = torch.from_numpy(gnd).long().to(args.device)

    # Autoencoder settings
    AE_layer_dims = [2048, 1024]
    AE_models = []
    optimizer_AE = []
    for i in range(view_num):
        AE_models.append(Autoencoder(AE_layer_dims, features[0][i].shape[1]).to(args.device))
        optimizer_AE.append(torch.optim.Adam(AE_models[-1].parameters(), lr = 0.001,betas=(0.90, 0.92), weight_decay=0.01))

    # FCNet settings
    input_dim = AE_layer_dims[-1]
    FCNet_layer_dims = [AE_layer_dims[-1] // 2, AE_layer_dims[-1]]
    FCnet_model = FCnet(N, input_dim, FCNet_layer_dims).to(args.device)
    for name, p in FCnet_model.named_parameters():
        if name == 'H':
            p.requires_grad = False
        else:
            p.requires_grad = True
    optimizer_FCNet_para = torch.optim.Adam(filter(lambda p: p.requires_grad, FCnet_model.parameters()),lr = 0.01, betas=(0.90, 0.92), weight_decay=0.01)
    for name, p in FCnet_model.named_parameters():
        if name == 'H':
            p.requires_grad = True
        else:
            p.requires_grad = False
    optimizer_FCNet_H = torch.optim.Adam(filter(lambda p: p.requires_grad, FCnet_model.parameters()),lr = 0.01, betas=(0.90, 0.92), weight_decay=0.01)
    

    # GCN settings
    dim_in = AE_layer_dims[-1]
    dim_out = num_class
    adj_hats = adj_hats.float().to(args.device)
    GCN_model = LGCN(adj_hats, dim_in, dim_out).to(args.device)
    optimizer_GCN = torch.optim.Adam(GCN_model.parameters(),lr = 0.01) 

    cross_entropy_loss = nn.CrossEntropyLoss()

    with tqdm(total=args.epoch_num, desc="Training") as pbar:
        for i in range(args.epoch_num):
            # Autoencoder training
            encode_list = []
            cur_loss_AE = 0.0
            for view_id in range(view_num):
                X = torch.from_numpy(features[0][view_id]).float().to(args.device)
                encode, decode = AE_models[view_id](X)
                encode_list.append(encode.detach())

                # BP for autoencoder
                loss_AE = norm_2(decode, X) + args.beta * AE_models[view_id].rho_loss(args.rho)
                cur_loss_AE += loss_AE
                optimizer_AE[view_id].zero_grad()
                loss_AE.backward()
                optimizer_AE[view_id].step()

            # FCNet training for parameters
            cur_loss_FC = 0.0
            for view_id in range(view_num):
                G = FCnet_model()
                loss_FC = norm_2(encode_list[view_id], G)
                cur_loss_FC += loss_FC
            # BP for FCNet parameters
            optimizer_FCNet_para.zero_grad()
            cur_loss_FC.backward()
            optimizer_FCNet_para.step()  

            # FCNet training for H
            cur_loss_H = 0.0
            for view_id in range(view_num):
                G = FCnet_model()
                loss_FC = norm_2(encode_list[view_id], G)
                cur_loss_H += loss_FC
            # BP for FCNet H
            optimizer_FCNet_H.zero_grad()
            cur_loss_H.backward()
            optimizer_FCNet_H.step()   

            # GCN Training
            GCN_input = FCnet_model.H.detach()
            y_pred = GCN_model(GCN_input)
            loss_GCN = cross_entropy_loss(y_pred[p_labeled], gnd[p_labeled])
            optimizer_GCN.zero_grad()
            loss_GCN.backward()
            optimizer_GCN.step()

            pbar.set_postfix({'Loss_AE' : '{0:1.5f}'.format(cur_loss_AE), 
                            'Loss_FCNet' : '{0:1.5f}'.format(cur_loss_H),
                            'Loss_GCN' : '{0:1.5f}'.format(loss_GCN)})
            pbar.update(1)
    
    pred_label = torch.argmax(F.log_softmax(y_pred,1), 1)
    accuracy_value = accuracy_score(gnd[p_unlabeled].cpu().detach().numpy(), pred_label[p_unlabeled].cpu().detach().numpy())
    print(args.dataset_name, "Accuracy:", accuracy_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run .")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--dataset-name", nargs = "?", default = "ALOI")
    parser.add_argument("--epoch-num", type = int, default = 180, help = "Number of training epochs.")
    parser.add_argument("--seed", type = int, default = 2023, help = "Random seed for network training.")
    parser.add_argument("--learning-rate", type = float, default = 0.01, help = "Learning rate. Default is 0.01.")

    parser.add_argument("--k", type = int, default = 15, help = "k of KNN graph.")
    parser.add_argument("--beta", type = float, default = 1.0, help = "beta. Default is 1.0")
    parser.add_argument("--rho", type = float, default = 0.05, help = "rho. Default is 0.05")
    

    args = parser.parse_args()

    print(args.dataset_name)
    test(args)
