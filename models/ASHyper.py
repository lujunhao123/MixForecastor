import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.data import data as D
from torch.nn import Linear
import torch_scatter
from math import sqrt
#from layers.Mis_Layers import EncoderLayer, Decoder, Predictor
#from layers.Mis_Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
# from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from layers.Embed import DataEmbedding, CustomEmbedding,DataEmbedding_new
from torch_geometric.utils import scatter
import math




class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        configs.device = torch.device("cuda")
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)


        self.all_size=get_mask(configs.seq_len, configs.window_size)
        self.Ms_length = sum(self.all_size)
        self.conv_layers = eval(configs.CSCM)(configs.enc_in, configs.window_size, configs.enc_in)
        self.out_tran = nn.Linear(self.Ms_length, self.pred_len)
        self.out_tran.weight=nn.Parameter((1/self.Ms_length)*torch.ones([self.pred_len,self.Ms_length]))
        self.chan_tran=nn.Linear(configs.d_model,configs.enc_in)
        self.inter_tran = nn.Linear(80, self.pred_len)
        self.concat_tra=nn.Linear(320,self.pred_len)

        self.dim=configs.d_model
        self.hyper_num=50
        self.embedhy=nn.Embedding(self.hyper_num,self.dim)
        self.embednod=nn.Embedding(self.Ms_length,self.dim)


        self.idx = torch.arange(self.hyper_num)
        self.nodidx=torch.arange(self.Ms_length)
        self.alpha=3
        self.k=10

        self.window_size=configs.window_size
        self.multiadphyper=multi_adaptive_hypergraoh(configs)
        self.hyper_num1 = configs.hyper_num
        self.hyconv=nn.ModuleList()
        self.hyperedge_atten=SelfAttentionLayer(configs)
        for i in range (len(self.hyper_num1)):
            self.hyconv.append(HypergraphConv(configs.enc_in, configs.enc_in))

        self.slicetran=nn.Linear(100,configs.pred_len)
        self.weight = nn.Parameter(torch.randn(self.pred_len, 76))

        self.argg = nn.ModuleList()
        for i in range(len(self.hyper_num1)):
            self.argg.append(nn.Linear(self.all_size[i],self.pred_len))
        self.chan_tran = nn.Linear(configs.enc_in, configs.enc_in)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # normalization
        mean_enc=x.mean(1,keepdim=True).detach()
        x=x - mean_enc
        std_enc=torch.sqrt(torch.var(x,dim=1,keepdim=True,unbiased=False)+1e-5).detach()
        x=x / std_enc
        adj_matrix = self.multiadphyper(x)
        seq_enc = self.conv_layers(x)

        sum_hyper_list = []
        for i in range(len(self.hyper_num1)):

            mask = torch.tensor(adj_matrix[i]).to(x.device)
            ###inter-scale
            node_value = seq_enc[i].permute(0,2,1)
            node_value = torch.tensor(node_value).to(x.device)
            edge_sums={}
            for edge_id, node_id in zip(mask[1], mask[0]):
                if edge_id not in edge_sums:
                    edge_id=edge_id.item()
                    node_id=node_id.item()
                    edge_sums[edge_id] = node_value[:, :, node_id]
                else:
                    edge_sums[edge_id] += node_value[:, :, node_id]


            for edge_id, sum_value in edge_sums.items():
                sum_value = sum_value.unsqueeze(1)
                sum_hyper_list.append(sum_value)


            ###intra-scale
            output,constrainloss = self.hyconv[i](seq_enc[i], mask)


            if i==0:
                result_tensor=output
                result_conloss=constrainloss
            else:
                result_tensor = torch.cat((result_tensor, output), dim=1)
                result_conloss+=constrainloss

        sum_hyper_list=torch.cat(sum_hyper_list,dim=1)
        sum_hyper_list=sum_hyper_list.to(x.device)
        padding_need=80-sum_hyper_list.size(1)
        hyperedge_attention=self.hyperedge_atten(sum_hyper_list)
        pad = torch.nn.functional.pad(hyperedge_attention, (0, 0, 0, padding_need, 0, 0))



        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:

            x = self.Linear(x.permute(0,2,1))

            x_out=self.out_tran(result_tensor.permute(0,2,1))###ori
            x_out_inter = self.inter_tran(pad.permute(0, 2, 1))

        x=x_out+x+x_out_inter
        x=self.Linear_Tran(x).permute(0,2,1)
        x = x * std_enc + mean_enc

        return x,result_conloss# [Batch, Output length, Channel]


class HypergraphConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.1,
                 bias=False):
        super(HypergraphConv, self).__init__(aggr='add')
        self.soft=nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention


        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels))

            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))

        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:

            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def __forward__(self,
                    x,
                    hyperedge_index,
                    alpha=None):

        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        # --------------------------------------------------------
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)

        return out

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j
        if alpha is not None:
            out=alpha.unsqueeze(-1)*out
        return out
    def forward(self, x, hyperedge_index):
        x = torch.matmul(x, self.weight)
        x1=x.transpose(0,1)
        x_i = torch.index_select(x1, dim=0, index=hyperedge_index[0])
        edge_sums = {}

        for edge_id, node_id in zip(hyperedge_index[1], hyperedge_index[0]):
            if edge_id not in edge_sums:
                edge_id = edge_id.item()
                node_id = node_id.item()
                edge_sums[edge_id] = x1[node_id, :, :]
            else:
                edge_sums[edge_id] += x1[node_id, :, :]
        result_list = torch.stack([value for value in edge_sums.values()], dim=0)
        x_j = torch.index_select(result_list, dim=0, index=hyperedge_index[1])
        loss_hyper = 0
        for k in range(len(edge_sums)):
            for m in range(len(edge_sums)):
                inner_product = torch.sum(edge_sums[k] * edge_sums[m], dim=1, keepdim=True)
                norm_q_i = torch.norm(edge_sums[k], dim=1, keepdim=True)
                norm_q_j = torch.norm(edge_sums[m], dim=1, keepdim=True)
                alpha = inner_product / (norm_q_i * norm_q_j)
                distan = torch.norm(edge_sums[k] - edge_sums[m],dim=1, keepdim=True)
                loss_item = alpha * distan + (1 - alpha) * (torch.clamp(torch.tensor(4.2) - distan, min=0.0))
                loss_hyper += torch.abs(torch.mean(loss_item))


        loss_hyper = loss_hyper / ((len(edge_sums) + 1)**2)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x1.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        D = degree(hyperedge_index[0], x1.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        B[B == float("inf")] = 0
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x1, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        out=out.transpose(0, 1)
        constrain_loss = x_i - x_j
        constrain_lossfin1=torch.mean(constrain_loss)
        constrain_losstotal = abs(constrain_lossfin1) + loss_hyper
        return out, constrain_losstotal
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class multi_adaptive_hypergraoh(nn.Module):
    def __init__(self,configs):
        super(multi_adaptive_hypergraoh, self).__init__()
        self.seq_len = configs.seq_len
        self.window_size=configs.window_size
        self.inner_size=configs.inner_size
        self.dim=configs.d_model
        self.hyper_num=configs.hyper_num
        self.alpha=3
        self.k=configs.k
        self.embedhy=nn.ModuleList()
        self.embednod=nn.ModuleList()
        self.linhy=nn.ModuleList()
        self.linnod=nn.ModuleList()
        for i in range(len(self.hyper_num)):
            self.embedhy.append(nn.Embedding(self.hyper_num[i],self.dim))
            self.linhy.append(nn.Linear(self.dim,self.dim))
            self.linnod.append(nn.Linear(self.dim,self.dim))
            if i==0:
                self.embednod.append(nn.Embedding(self.seq_len,self.dim))
            else:
                product=math.prod(self.window_size[:i])
                layer_size=math.floor(self.seq_len/product)
                self.embednod.append(nn.Embedding(int(layer_size),self.dim))

        self.dropout = nn.Dropout(p=0.1)


    def forward(self,x):
        node_num = []
        node_num.append(self.seq_len)
        for i in range(len(self.window_size)):
            layer_size = math.floor(node_num[i] / self.window_size[i])
            node_num.append(layer_size)
        hyperedge_all=[]

        for i in range(len(self.hyper_num)):
            hypidxc=torch.arange(self.hyper_num[i]).to(x.device)
            nodeidx=torch.arange(node_num[i]).to(x.device)
            hyperen=self.embedhy[i](hypidxc)
            nodeec=self.embednod[i](nodeidx)

            a = torch.mm(nodeec, hyperen.transpose(1, 0))
            adj=F.softmax(F.relu(self.alpha*a))
            mask = torch.zeros(nodeec.size(0), hyperen.size(0)).to(x.device)
            mask.fill_(float('0'))
            s1, t1 = adj.topk(min(adj.size(1),self.k), 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask
            adj = torch.where(adj > 0.5, torch.tensor(1).to(x.device), torch.tensor(0).to(x.device))
            adj = adj[:, (adj != 0).any(dim=0)]
            matrix_array = torch.tensor(adj, dtype=torch.int)
            result_list = [list(torch.nonzero(matrix_array[:, col]).flatten().tolist()) for col in
                           range(matrix_array.shape[1])]

            node_list = torch.cat([torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]).tolist()
            count_list = list(torch.sum(adj, dim=0).tolist())
            hperedge_list = torch.cat([torch.full((count,), idx) for idx, count in enumerate(count_list, start=0)]).tolist()
            hypergraph=np.vstack((node_list,hperedge_list))
            hyperedge_all.append(hypergraph)

        return hyperedge_all



class SelfAttentionLayer(nn.Module):
    def __init__(self, configs):
        super(SelfAttentionLayer, self).__init__()
        self.query_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.key_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.value_weight = nn.Linear(configs.enc_in, configs.enc_in)

    def forward(self, x):
        q = self.query_weight(x)
        k = self.key_weight(x)
        v = self.value_weight(x)
        attention_scores = F.softmax(torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=-1)
        attended_values = torch.matmul(attention_scores, v)

        return attended_values

def get_mask(input_size, window_size):
    """Get the attention mask of HyperGraphConv"""
    # Get the size of all layers
    # window_size=[4,4,4]
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)
    return all_size