import pandas as pd
import torch
import numpy as np
import os
import random


class ProtoNetText(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, proto_dim):
        super(ProtoNetText, self).__init__()
        self.embed_size = embedding_size
        self.hidden_size = hidden_size
        self.proto_dim = proto_dim
        self.l1 = torch.nn.Linear(self.embed_size, self.hidden_size)
        self.rep_block =torch.nn.Sequential(*[torch.nn.BatchNorm1d(hidden_size), torch.nn.Linear(self.hidden_size, self.hidden_size)])
        self.final = torch.nn.Linear(self.hidden_size, self.proto_dim)
    def forward(self, x):
        return self.final(self.rep_block(self.l1(x)))
    
# x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries
class ProtoLoss(torch.nn.Module):
    def __init__(self, num_classes, num_support, num_queries, ndim):
        super(ProtoLoss,self).__init__()
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_queries = num_queries
        self.ndim = ndim
    
    def euclidean_distance(self, a, b):
        # a.shape = N x D
        # b.shape = M x D
        N, D = a.shape[0], a.shape[1]
        M = b.shape[0]
        a = torch.repeat_interleave(a.unsqueeze(1), repeats = M, dim = 1)
        b = torch.repeat_interleave(b.unsqueeze(0), repeats = N, dim = 0)
        return 1.*torch.sum(torch.pow((a-b), 2),2)
        
    def forward(self, x, q, labels_onehot):
        protox = torch.mean(1.*x.reshape([self.num_classes,self.num_support,self.ndim]),1)
        dists = self.euclidean_distance(protox, q)
        logpy = torch.log_softmax(-1.*dists,0).transpose(1,0).view(self.num_classes,self.num_queries,self.num_classes)
        ce_loss = -1. * torch.mean(torch.mean(logpy * labels_onehot.float(),1))
        accuracy = torch.mean((torch.argmax(labels_onehot.float(),-1).float() == torch.argmax(logpy,-1).float()).float())
        return ce_loss, accuracy