import torch
import torch.nn as nn
from typing import Union
import copy
import pdb
from config import cfg
import torch.nn.functional as F

class Combiner(nn.Module):
    def __init__(self,
                 total_num_selects: int,
                 num_classes: int,
                 inputs: Union[dict, None] = None,
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):

        super(Combiner, self).__init__()
        self.fpn_size = fpn_size
        if fpn_size is None:
            for name in inputs:
                if len(name) == 4:
                    in_size = inputs[name].size(1)
                elif len(name) == 3:
                    in_size = inputs[name].size(2)
                else:
                    raise ValueError("The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_" + name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        # Componets for adaptive graph builder
        num_joints = total_num_selects // 32
        A = torch.eye(num_joints) / 100 + 1 / 100
        self.base_adjacency = nn.Parameter(copy.deepcopy(A))
        self.input_proj = nn.Linear(total_num_selects, num_joints)
        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size // 4, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size // 4, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

        
        # Accually equals to a NxN weight matrix
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)

        self.lin = nn.Linear(self.proj_size, self.proj_size)
        self.lin2 = nn.Linear(self.proj_size, self.proj_size)

        # merge information
        self.output_proj = nn.Linear(num_joints, 1)

        # Claasifier: class predict
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(self.proj_size, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):

        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_"+name)(x[name]))
            else:
                hs.append(x[name])
        # hs: [N x (sum_of_select_features) x feature_dim] -> [N x feature_dim x sum]
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous()  
  
        # Project hs to [N x feature_dim x sum // 32]
        hs = self.input_proj(hs)

        # Adaptively build the adjacency matrix, Build a graph with (sum // 32) nodes, here is 85
        # conv1d : [N x in_feature_dim x node_num] -> [N x out_feature_dim x node_num]
        q1 = self.conv_q1(hs).mean(1)
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.base_adjacency + A1 * self.alpha
        
        if cfg.model.positive_adj == 'abs':
            A1 = torch.abs(A1)
  
        # Graph convolution
        
        if cfg.model.combiner == 'original':
            # Accually equals to a NxN weight matrix
            # 2 x 1536 x 85 ->  2 x 1536 x 85
            hs = self.conv1(hs) 
            hs = torch.matmul(hs, A1)
        elif cfg.model.combiner == 'new':
            # pdb.set_trace()
            hs = torch.transpose(hs, 1, 2)
            hs = self.lin(hs)
            hs = torch.bmm(A1, hs)

            hs = torch.transpose(hs, 1, 2)

        elif cfg.model.combiner == 'new-2hop':
            hs = torch.transpose(hs, 1, 2)
            hs = self.lin(hs)
            hs = torch.bmm(A1, hs)
            hs = F.relu(hs)
            hs = self.lin2(hs)
            hs = torch.bmm(A1, hs)
            
            hs = torch.transpose(hs, 1, 2)

        hs = self.batch_norm1(hs)
        # predict
        hs = self.output_proj(hs)
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.fc(hs)

        return hs 


# 比较低效的实现方法，也不够好。期待从这个角度进行改进。
class GCNCombiner(nn.Module):

    def __init__(self,
                 total_num_selects: int,
                 num_classes: int,
                 inputs: Union[dict, None] = None,
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None,
                 positive_adj: bool = False):
        """
        If building backbone without FPN, set fpn_size to None and MUST give 
        'inputs' and 'proj_size', the reason of these setting is to constrain the 
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()

        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        # auto-proj
        self.fpn_size = fpn_size
        if fpn_size is None:
            for name in inputs:
                if len(name) == 4:
                    in_size = inputs[name].size(1)
                elif len(name) == 3:
                    in_size = inputs[name].size(2)
                else:
                    raise ValueError(
                        "The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_"+name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        # build one layer structure (with adaptive module)
        num_joints = total_num_selects // 32

        self.param_pool0 = nn.Linear(total_num_selects, num_joints)

        self.positive_adj = positive_adj

        A = torch.eye(num_joints)/100 + 1/100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)

        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        # merge information
        self.param_pool1 = nn.Linear(num_joints, 1)

        # class predict
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.proj_size, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_"+name)(x[name]))
            else:
                hs.append(x[name])
        # hs: [N x (sum_of_select_features) x feature_dim] -> [N x feature_dim x sum]
        hs = torch.cat(hs, dim=1).transpose(
            1, 2).contiguous()  # B, S', C --> B, C, S
  
        # Project hs to [N x feature_dim x sum // 32]
        hs = self.param_pool0(hs)
        # adaptive adjacency
        # Build a graph with (sum // 32) nodes, here is 85
        # conv1d : [N x in_feature_dim x node_num] -> [N x out_feature_dim x node_num]
        q1 = self.conv_q1(hs).mean(1)
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1
        
        if cfg.model.positive_adj == 'abs':
            A1 = torch.abs(A1)
        # elif cfg.model.positive_adj == 'exp':
        #     A1 = torch.exp(A1)
        #     A1 = torch.softmax(A1, dim=0)

        # graph convolution
        hs = self.conv1(hs)
        hs = torch.matmul(hs, A1)

        hs = self.batch_norm1(hs)
        # predict
        hs = self.param_pool1(hs)
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs
