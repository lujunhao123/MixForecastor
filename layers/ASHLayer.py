from torch.functional import align_tensors
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch
import math
import numpy as np
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import data as D
from typing import Optional
from torch import Tensor
from Embed import *


class ScaledDotProductAttention1(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.hatt1 = HypergraphConv(32, 1)
        self.hatt2 = HypergraphConv(1, 32)
        self.liear = torch.nn.Linear(32, 1)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # if mask is not None:
        #     attn = attn.masked_fill(mask, -1e9)
        # x = F.leaky_relu(self.hatt1(attn, mask), 0.2)
        # x = F.leaky_relu(self.hatt2(x, mask), 0.2)
        # attn = F.leaky_relu(self.liear(x))
        # output = torch.matmul(attn, v)


        x = self.hatt1(attn, mask)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.hatt2(x, mask)
        attn = F.leaky_relu(self.liear(x))
        # print(attn.shape)
        # output = torch.matmul(attn, v)
        return v, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):##temperature=11.3137084对于128维度进行了根号d
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(dataset_cora.num_node_features, 168)
        # self.conv2 = GCNConv(168, dataset_cora.num_classes)
        # self.conv1 = HypergraphConv(1, 16)
        # self.conv2 = HypergraphConv(16, 2)
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x

class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W}` is the diagonal hyperedge weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=False,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(HypergraphConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
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
                    hyperedge_weight=None,
                    alpha=None):

        if hyperedge_weight is None:
            D = degree(hyperedge_index[0], x.size(0), x.dtype)
        else:
            D_1 = scatter_add(
                hyperedge_weight[hyperedge_index[1, 0:13263]],
                hyperedge_index[0, 0:13263],
                dim=0,
                dim_size=x.size(0))
            D_2 = scatter_add(
                hyperedge_weight[hyperedge_index[1, 13264:112859]],
                hyperedge_index[0, 13264:112859],
                dim=0,
                dim_size=x.size(0))
            D = torch.cat((D_1, D_2), dim=0)
            # ---------------------------------------------------------
        D = 1.0 / D
        D[D == float("inf")] = 0


        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B_1 = 1.0 / degree(hyperedge_index[1, 0:13263], 2708, x.dtype)
        B_2 = 1.0 / degree(hyperedge_index[1, 13264:], int(num_edges/2), x.dtype)
        B = torch.cat((B_1, B_2), dim=0)
        # ---------------------------------------------------------

        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            # B = B * hyperedge_weight; two next line is added by myself
            B = B * hyperedge_weight.t()
            B = B.t()

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        return out

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j.view(
            -1, self.heads, self.out_channels)
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyper_edge_index (LongTensor): Hyperedge indices from
                :math:`\mathbf{H}`.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights from
                :math:`\mathbf{W}`. (default: :obj:`None`)
        """
        x = torch.matmul(x, self.weight)
        # hyperedge_weight=data.edge_attr
        alpha = None

        if self.use_attention:
            x = x.view(-1, self.heads, self.out_channels)
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.__forward__(x, hyperedge_index, hyperedge_weight, alpha)

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
    



class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)##[input 512,output 768]
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head  ###d_k,d_v=128 n_head=6
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)###sz_b=32, len_q=223, len_k=223, len_v=223

        residual = q   #32 223 512
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) #[32,223,6,128]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) #[32,223,6,128]
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) #[32,223,6,128]

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)###q,k,v=#[32,6,223,128]

        if mask is not None: ###mask[32,223,223]
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)  # For head axis broadcasting.###[32,1,223,223]为了和q,k,v维度对齐

        output, attn = self.attention(q, k, v, mask=mask)# attn=[32,6,223,223]    output=[32,6,223,128]
        # print(output.shape)   #[32,6,223,128]

        # Transpose to move the head dimension back: b x lq x n x dv  32x223x6x128
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1) ###[32,223,6x128]
        output = self.dropout(self.fc(output))###[32,223,512]
        output += residual###[32,223,512]

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        #self.layer_norm = GraphNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


def get_mask(input_size, window_size, inner_size, device):
    """Get the attention mask of HyperGraphConv"""
    # Get the size of all layers
    # window_size=[4,4,4]
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)#####all_size=[169,7,1]

    window_size1 = [24, 7]
    all_size1 = []
    all_size1.append(input_size)
    for i in range(len(window_size1)):
        layer_size1 = math.floor(all_size1[i] / window_size1[i])
        all_size1.append(layer_size1)


    window_sizelen=4
    seq_length = sum(all_size)###[177]
    small = [int(i) for i in range(0, seq_length)]
    mask_edge=small
    small1=[int(i) for i in range(0, seq_length)]

    """
    for i in range(seq_length):
        if i<=seq_length/window_sizelen:
            mask_edge[i]=0
        elif i>seq_length/window_sizelen and i<=(seq_length/window_sizelen+seq_length/window_sizelen**2):
            mask_edge[i]=1
        elif i>(seq_length/window_sizelen+seq_length/window_sizelen**2) and i<=(seq_length/window_sizelen+seq_length/window_sizelen**3):
            mask_edge[i]=2
        else:
            mask_edge[i]=3
    """
    num_flag=1
    time_flag=24
    # 初始化包含未知数量点的列表
    total_points = all_size[0]
    points = [0] * total_points

    # 定义每个区间的长度
    interval_length = 24

    # 循环赋值
    value = 1
    for i in range(0, total_points, interval_length):
        end_index = min(i + interval_length, total_points)
        mask_edge[i:end_index] = [value] * (end_index - i)
        value += 1

    # 打印结果
    # print(points)

    # for i in range(seq_length):
    #     # if i <= all_size[0]:
    #     #     # if i %
    #     #     mask_edge[i] = 0
    #     if i > all_size[0] and i <= (all_size[0]+all_size[1]):
    #         mask_edge[i] = 9
    #     elif i > all_size[1] and i <= (all_size[0]+all_size[1]+all_size[2]):
    #         mask_edge[i] = 10
    #     elif i > (all_size[0]+all_size[1]+all_size[2]):
    #         mask_edge[i] = 11
    # edge_index_test=np.vstack((small1,mask_edge))
    # edge_index_test = torch.tensor(edge_index_test, dtype=torch.long)








    for i in range(seq_length):
        if i <= all_size[0]:
            mask_edge[i] = 0
        elif i > all_size[0] and i <= (all_size[0]+all_size[1]):
            mask_edge[i] = 1
        elif i > all_size[1] and i <= (all_size[0]+all_size[1]+all_size[2]):
            mask_edge[i] = 2
        else:
            mask_edge[i] = 3
    edge_index=np.vstack((small1,mask_edge))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    s = 1
    h = 3
    a = []
    number_edge = []
    j = 0
    for i in range(1,int(seq_length/h)):
        base = (i - 1) // (s + 1)
        horizon = (s + 1) * h
        bias = (i - 1) % (s + 1) + 1
        result = base * horizon + bias
        for j in range(h):
            # num=list(range)
            a.append(result)
            # print(result, end=" ")
            result = result + s + 1
        edge_index1 = list(np.repeat(j, h))
        number_edge += edge_index1
        j += 1
        # print()
    intra_resultsk = np.vstack((a, number_edge))
    intra_resultsk = torch.tensor(intra_resultsk, dtype=torch.long)
    # print(intra_resultsk)
    """
    # 实验结果显示，intra_result的效果不是很好，说明随意搭建超边不可取
    i=0
    j=0
    bbb = 0
    intra_all = []
    num_all=[]
    for layer_idx in range(1,len(all_size1)):
        start = sum(all_size1[:layer_idx])
        begin=sum(all_size1[:layer_idx-1])
        aaa=begin+all_size1[layer_idx-1]

        while (start-bbb>0 ) :
            bbb+=window_size1[layer_idx-1]
            bbb=min(bbb,aaa)
            left_side=max(begin,i*window_size1[layer_idx-1])
            right_side = min(bbb, aaa)
            # right_side=min((i+1)*window_size[layer_idx-1],aaa)
            # num=list(range((i-1)*window_size[layer_idx-1],right_side))
            num = list(range(left_side, right_side))
            num_all += num
            intra_edge = list(np.repeat(j, len(num)))
            intra_all += intra_edge
            i+=1
            j+=1
    intra_result2 = np.vstack((num_all, intra_all))
    intra_result2 = torch.tensor(intra_result2, dtype=torch.long)
    # mask = torch.zeros(seq_length, seq_length, device=device)
    """


    # edge_index1test
    s = 3
    h = 4
    num_all = []
    intra_all = []
    k = 0
    for i in range(0, seq_length+1):
        base = (i - 1) // (s + 1)
        horizon = (s + 1) * h
        bias = (i - 1) % (s + 1) + 1
        result = base * horizon + bias
        if i % h == 0 and i > 0:
            num = list(range(i - h, i))
            intra_edge = list(np.repeat(k, h))
            k += 1
            # print(k)
            num_all += num
            intra_all += intra_edge
    edge_index1 = np.vstack((num_all, intra_all))
    intra_result2 = torch.tensor(edge_index1, dtype=torch.long)


    """
    intra_result3的效果也不是很好，搭配edge_index效果稍微好点，但是不如直接用edge_index
    i=0
    j=4
    bbb = 0
    intra_all = []
    num_all=[]
    # #######get_intra test
    for layer_idx in range(1, 2):
        start = sum(all_size1[:layer_idx])
        begin = sum(all_size1[:layer_idx - 1])
        aaa = begin + all_size1[layer_idx - 1]
        week_index=list(range(sum(all_size1[:1]),sum(all_size1[:2])))

        while (start - bbb > 0):
            bbb += window_size1[layer_idx - 1]
            bbb = min(bbb, aaa)
            left_side = max(begin, i * window_size1[layer_idx - 1])
            right_side = min(bbb, aaa)
            # right_side=min((i+1)*window_size[layer_idx-1],aaa)
            # num=list(range((i-1)*window_size[layer_idx-1],right_side))
            num = list(range(left_side, right_side))
            num+=week_index
            num_all += num
            intra_edge = list(np.repeat(j, len(num)))
            intra_all += intra_edge
            i += 1
            j += 1
    intra_result3 = np.vstack((num_all, intra_all))
    # intra_result3 = torch.tensor(intra_result3, dtype=torch.long)
    """


    """"
    #效果较差
    j=4
    intra_all=[]
    num_all=[]
    inner_size1=4
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):###左边选1个右边选两个，加上自身共四个
            if i % inner_size1==0 and i !=0:
                left_side = max(i - inner_size1, start)
                num = list(range(left_side, i))
                num_all += num
                intra_edge = list(np.repeat(j, len(num)))
                intra_all += intra_edge
                j+=1
                k=i
        if(start + all_size[layer_idx]!=k):
            num=list(range(k, start + all_size[layer_idx]))
            num_all += num
            intra_edge = list(np.repeat(j, len(num)))
            intra_all += intra_edge
            j += 1
    intra_result4 = np.vstack((num_all, intra_all))
    # intra_result4 = torch.tensor(intra_result4, dtype=torch.long)

    """

    """
    #效果较差
    # i=0
    j=0
    bbb = 0
    intra_all1 = []
    num_all1=[]
    inner_window = inner_size // 2
    for layer_idx in range(1,len(all_size)+1):
        start = sum(all_size[:layer_idx])
        begin = sum(all_size[:layer_idx - 1])
        aaa = begin + all_size[layer_idx - 1]
        i=begin
        bbb=sum(all_size[:layer_idx-1])
        while (start - bbb > 0):
            bbb += inner_size
            bbb = min(bbb, aaa)
            left_side = max(begin, i)
            right_side = min(bbb, aaa)
            # right_side=min((i+1)*window_size[layer_idx-1],aaa)
            # num=list(range((i-1)*window_size[layer_idx-1],right_side))
            num = list(range(left_side, right_side))
            num_all1 += num
            intra_edge = list(np.repeat(j, len(num)))
            intra_all1 += intra_edge
            # i += 1
            i = i + inner_size
            j += 1
    intra_result5 = np.vstack((num_all1, intra_all1))
    intra_result5 = torch.tensor(intra_result5, dtype=torch.long)
    """




    """
    ori
    j=0
    intra_all=[]
    num_all=[]
    # inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            if (i+1) % 4 == 0 or (i+1 == start + all_size[layer_idx]):
                left_side = max(i - inner_size, start)
                right_side = min(i+1 , start + all_size[layer_idx])
                num = list(range(left_side, right_side))
                num_all += num
                intra_edge = list(np.repeat(j, len(num)))
                intra_all += intra_edge
                j += 1
    intra_result6 = np.vstack((num_all, intra_all))
    intra_result6 = torch.tensor(intra_result6, dtype=torch.long)
    """
    """
    #混合尺度
    j=0
    intra_all=[]
    num_all=[]
    # inner_window = inner_size // 2
    # mix_scale
    mixnum_all=[]
    mix_all=[]
    for i in range(1,all_size[0]+1):
        # print(i)
        if (i%h==0 and i>0) or (i==(all_size[0])):
            left_side=i-h
            right_side=min(i,all_size[0]+1)
            num=list(range(left_side,right_side))
            mixnum_all+=num
            node_scale1=math.ceil(left_side/sum(window_size[:1]))+sum(all_size[:1])+1
            mixnum_all.append(node_scale1)
            node_scale2=math.ceil(left_side/(window_size[0]*window_size[1]))+sum(all_size[:2])
            node_scale2=min(node_scale2,sum(all_size[:3]))
            mixnum_all.append(node_scale2)
            node_scale3 = math.ceil(left_side / (window_size[0]*window_size[1]*window_size[2])) + sum(all_size[:3])
            node_scale3=min(node_scale3,sum(all_size[:4]))
            mixnum_all.append(node_scale3)
            mix_edge = list(np.repeat(j, len(num)+3))
            # for s in range(len(window_size)):
            #     node_scale=math.ceil(right_side/window_size[s])
            mix_all += mix_edge
            j += 1
    mix_result = np.vstack((mixnum_all, mix_all))
    mix_result = torch.tensor(mix_result, dtype=torch.long)
    a=mixnum_all
    """

    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            if (i+1) % 4 == 0 or (i+1 == start + all_size[layer_idx]):
                left_side = max(i - inner_size, start)
                right_side = min(i+1 , start + all_size[layer_idx])
                num = list(range(left_side, right_side))

                num_all += num
                intra_edge = list(np.repeat(j, len(num)))
                intra_all += intra_edge
                j += 1
    intra_result6 = np.vstack((num_all, intra_all))
    # print(intra_result6)
    intra_result6 = torch.tensor(intra_result6, dtype=torch.long)



    """
    # get intra-scale mask
    j=4
    intra_all=[]
    num_all=[]
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            # mask[i, left_side:right_side] = 1
            num = list(range(left_side, right_side))
            num_all += num
            intra_edge = list(np.repeat(j, len(num)))
            intra_all += intra_edge
            j += 1
    intra_result7 = np.vstack((num_all, intra_all))
    intra_result7 = torch.tensor(intra_result7, dtype=torch.long)
    """

    """
    # get inter-scale mask
    j = 0
    num_all = []
    index_all = []
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):  ###第一层[169,211]
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[layer_idx - 1]
            if i == (start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]
            #####得到尺度间数据列表
            num = list(range(left_side, right_side))
            num += list([i])
            num_all += num
            #####得到尺度间数据边的索引
            index = list(np.repeat(j, len(num)))
            index_all += index
            # print(mask[[5],:])#####输出第五行
            j += 1
            # mask[i, left_side:right_side] = j
            # mask[left_side:right_side, i] = j####第0行到第4行为同一个父节点i
    edge_inter = np.vstack((num_all, index_all))
    edge_inter = torch.tensor(edge_inter, dtype=torch.long)
    """

    # edge_index=np.hstack((edge_index,edge_inter))
    # edge_index = np.hstack((edge_index, mix_result))
    # edge_index = np.hstack((edge_index, intra_result6))
    # edge_index = torch.tensor(edge_index, dtype=torch.long)

    # edge_index1 = torch.tensor(edge_index1, dtype=torch.long)
    # edge_index = np.hstack((edge_index1, intra_result6))

    # edge_index = np.hstack((edge_index, intra_result6))
    # edge_index = torch.tensor(edge_index, dtype=torch.long)


    # mask


    # return edge_index, all_size
    # edge_index_test
    return intra_result6, all_size
    # return edge_index_test, all_size






def refer_points(all_sizes, window_size, device):
    """Gather features from PAM's pyramid sequences"""
    input_size = all_sizes[0]###all_size=[169,42,10,2]
    indexes = torch.zeros(input_size, len(all_sizes), device=device)###index=[169,4]

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            indexes[i][j] = former_index

    indexes = indexes.unsqueeze(0).unsqueeze(3)###unsqueeze(0)在0这个维度增加一维

    return indexes.long()


def get_subsequent_mask(input_size, window_size, predict_step, truncate):
    """Get causal attention mask for decoder."""
    if truncate:
        mask = torch.zeros(predict_step, input_size + predict_step)
        for i in range(predict_step):
            mask[i][:input_size+i+1] = 1
        mask = (1 - mask).bool().unsqueeze(0)
    else:
        all_size = []
        all_size.append(input_size)
        for i in range(len(window_size)):
            layer_size = math.floor(all_size[i] / window_size[i])
            all_size.append(layer_size)
        all_size = sum(all_size)
        mask = torch.zeros(predict_step, all_size + predict_step)
        for i in range(predict_step):
            mask[i][:all_size+i+1] = 1
        mask = (1 - mask).bool().unsqueeze(0)

    return mask


def get_q_k(input_size, window_size, stride, device):
    """
    Get the index of the key that a given query needs to attend to.
    """
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)

    max_attn += window_size + 1
    mask = torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1

    for i in range(input_size):
        mask[i, 0:window_size] = i + torch.arange(window_size) - window_size // 2
        mask[i, mask[i] > input_size - 1] = -1

        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    for i in range(second_length):
        mask[input_size+i, 0:window_size] = input_size + i + torch.arange(window_size) - window_size // 2
        mask[input_size+i, mask[input_size+i] < input_size] = -1
        mask[input_size+i, mask[input_size+i] > third_start - 1] = -1

        if i < second_length - 1:
            mask[input_size+i, window_size:(window_size+stride)] = torch.arange(stride) + i * stride
        else:
            mask[input_size+i, window_size:(window_size+second_last)] = torch.arange(second_last) + i * stride

        mask[input_size+i, -1] = i // stride + third_start
        mask[input_size+i, mask[input_size+i] > fourth_start - 1] = fourth_start - 1
    for i in range(third_length):
        mask[third_start+i, 0:window_size] = third_start + i + torch.arange(window_size) - window_size // 2
        mask[third_start+i, mask[third_start+i] < third_start] = -1
        mask[third_start+i, mask[third_start+i] > fourth_start - 1] = -1

        if i < third_length - 1:
            mask[third_start+i, window_size:(window_size+stride)] = input_size + torch.arange(stride) + i * stride
        else:
            mask[third_start+i, window_size:(window_size+third_last)] = input_size + torch.arange(third_last) + i * stride

        mask[third_start+i, -1] = i // stride + fourth_start
        mask[third_start+i, mask[third_start+i] > full_length - 1] = full_length - 1
    for i in range(fourth_length):
        mask[fourth_start+i, 0:window_size] = fourth_start + i + torch.arange(window_size) - window_size // 2
        mask[fourth_start+i, mask[fourth_start+i] < fourth_start] = -1
        mask[fourth_start+i, mask[fourth_start+i] > full_length - 1] = -1

        if i < fourth_length - 1:
            mask[fourth_start+i, window_size:(window_size+stride)] = third_start + torch.arange(stride) + i * stride
        else:
            mask[fourth_start+i, window_size:(window_size+fourth_last)] = third_start + torch.arange(fourth_last) + i * stride

    return mask


def get_k_q(q_k_mask):
    """
    Get the index of the query that can attend to the given key.
    """
    k_q_mask = q_k_mask.clone()
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            if q_k_mask[i, j] >= 0:
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] ==i )[0]

    return k_q_mask


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True, use_tvm=False, q_k_mask=None, k_q_mask=None):
        super(EncoderLayer, self).__init__()
        self.use_tvm = use_tvm
        if use_tvm:
            from .PAM_TVM import PyramidalAttention
            self.slf_attn = PyramidalAttention(n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before, q_k_mask=q_k_mask, k_q_mask=k_q_mask)
            ###n_head=6, d_model=512, d_k=128, d_v=128
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, slf_attn_mask=None):
        if self.use_tvm:
            enc_output = self.slf_attn(enc_input)
            enc_slf_attn = None
        else:
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)##enc_input[32,223,512] mask[32,223,223]

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, Q, K, V, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            Q, K, V, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)####input_x[32,128,42]
        x = self.norm(x)
        x = self.activation(x)
        return x


class Conv_Constructnew(nn.Module):
    """Convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Conv_Construct, self).__init__()
        # if not isinstance(window_size, list):
        #     self.conv_layers = nn.ModuleList([
        #         ConvLayer(d_model, window_size),
        #         ConvLayer(d_model, window_size),
        #         ConvLayer(d_model, window_size)
        #         ])
        # else:
        #     self.conv_layers = nn.ModuleList([
        #         ConvLayer(d_model, window_size[0]),
        #         ConvLayer(d_model, window_size[1]),
        #         ConvLayer(d_model, window_size[2])
        #         ])
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size)
                # ConvLayer(d_model, window_size)
            ])
        else:
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size[0]),
                ConvLayer(d_model, window_size[1])
                # ConvLayer(d_model, window_size[2])
                ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)

        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs
class Conv_Construct(nn.Module):
    """Convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Conv_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size)
                ])
        else:
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size[0]),
                ConvLayer(d_model, window_size[1]),
                ConvLayer(d_model, window_size[2])
                ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)

        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs

class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size)
                ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = Linear(d_inner, d_model)####d_inner128 d_model=512
        self.down = Linear(d_model, d_inner)####d_model=512 d_inner128
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):####[32,169,512]

        temp_input = self.down(enc_input).permute(0, 2, 1)####先下采样，变为[32,169,128],再交换第1维度和第二维度-->[32,128,169]
        all_inputs = []
        all_inputs.append(temp_input.permute(0, 2, 1))
        ####对169个节点进行卷积
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)####第一次[32,128,42]第二次[32,128,10]第三次[32,128,2]
            all_inputs.append(temp_input.permute(0, 2, 1))
        """
        # print(all_inputs[1].shape())
        # hour_trend=self.up(all_inputs[0].transpose(1, 2))
        # day_trend=self.up(all_inputs[1].transpose(1, 2))
        # week_trend=self.up(all_inputs[2].transpose(1, 2))
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)####[32,54,128]
        all_inputs = self.up(all_inputs)####[32,54,512]
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)####enc_input[32,169,512]all_input=[32,223,512]

        all_inputs = self.norm(all_inputs)####[32,223,512]
        # hour_trend = self.norm(hour_trend)
        # day_trend = self.norm(day_trend)
        # week_trend = self.norm(week_trend)
        """
        return all_inputs





'''
class Scale_Construct(nn.Module):
    """Bottleneck convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Scale_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size)
                ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
            self.dropout = nn.Dropout(0.1)


        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_inner, kernel_size=4, stride=4, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, kernel_size=4, stride=4, bias=False)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_inner, kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, kernel_size=4, stride=4, bias=False)
        self.activation = F.relu
        self.up_trend = nn.Conv1d(in_channels=d_inner, out_channels=d_model, kernel_size=1, bias=False)






        self.up = Linear(d_inner, d_model)####d_inner128 d_model=512
        self.down = Linear(d_model, d_inner)####d_model=512 d_inner128
        self.norm = nn.LayerNorm(d_model)
        self.hour_trend=Linear(42, 169)
        self.day_trend=Linear(10, 169)
        self.week_trend = Linear(2, 169)

        self.all_trend = Linear(223,169)

    def forward(self, enc_input):####[32,169,512]
        hour_trend = self.dropout(self.activation(self.conv3(enc_input.transpose(-1, 1))))
        day_trend=self.dropout(self.activation(self.conv1(enc_input.transpose(-1, 1))))
        week_trend=self.dropout(self.conv2(day_trend))
        month_trend=self.dropout(self.conv4(week_trend))
        temp_input = self.down(enc_input).permute(0, 2, 1)####先下采样，变为[32,169,128],再交换第1维度和第二维度-->[32,128,169]
        all_inputs = []
        ####对169个节点进行卷积
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)####第一次[32,128,42]第二次[32,128,10]第三次[32,128,2]
            all_inputs.append(temp_input)


        # print(all_inputs[1].shape())
        # hour_trend1 = self.hour_trend(day_trend)
        # day_trend = self.day_trend(all_inputs[1])
        # week_trend = self.week_trend(all_inputs[2])
        hour_trend=self.up_trend(hour_trend).transpose(1, 2)
        day_trend=self.up_trend(day_trend).transpose(1, 2)
        week_trend=self.up_trend(week_trend).transpose(1, 2)
        month_trend=self.up_trend(month_trend).transpose(1, 2)
        # hour_trend=self.hour_trend(hour_trend)
        # day_trend=self.hour_trend(day_trend)
        # week_trend = self.week_trend(day_trend)
        all_trend=torch.cat([hour_trend,day_trend,week_trend,month_trend],dim=1)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)####[32,54,128]
        all_inputs = self.up(all_inputs)####[32,54,512]
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)####enc_input[32,169,512]all_input=[32,223,512]
        # all_trend=self.all_trend(all_trend.transpose(1, 2)).transpose(1, 2)

        all_inputs = self.norm(all_inputs)####[32,223,512]
        hour_trend = self.norm(hour_trend)
        day_trend = self.norm(day_trend)
        week_trend = self.norm(week_trend)
        all_trend = self.norm(all_trend)

        return all_inputs,hour_trend,day_trend,week_trend,all_trend
'''



class MaxPooling_Construct(nn.Module):
    """Max pooling CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(MaxPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList([
                nn.MaxPool1d(kernel_size=window_size),
                nn.MaxPool1d(kernel_size=window_size),
                nn.MaxPool1d(kernel_size=window_size)
                ])
        else:
            self.pooling_layers = nn.ModuleList([
                nn.MaxPool1d(kernel_size=window_size[0]),
                nn.MaxPool1d(kernel_size=window_size[1])
                ])
            # self.pooling_layers = nn.ModuleList([
            #     nn.MaxPool1d(kernel_size=window_size[0]),
            #     nn.MaxPool1d(kernel_size=window_size[1]),
            #     nn.MaxPool1d(kernel_size=window_size[2])
            #     ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        # all_inputs.append(enc_input)
        all_inputs.append(enc_input.transpose(1, 2))

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            # all_inputs.append(enc_input)
            all_inputs.append(enc_input.transpose(1, 2))

        # all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        # all_inputs = self.norm(all_inputs)

        return all_inputs


class AvgPooling_Construct(nn.Module):
    """Average pooling CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(AvgPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList([
                nn.AvgPool1d(kernel_size=window_size),
                nn.AvgPool1d(kernel_size=window_size),
                nn.AvgPool1d(kernel_size=window_size)
                ])
        else:
            self.pooling_layers = nn.ModuleList([
                nn.MaxPool1d(kernel_size=window_size[0]),
                nn.MaxPool1d(kernel_size=window_size[1])
                ])
            # self.pooling_layers = nn.ModuleList([
            #     nn.AvgPool1d(kernel_size=window_size[0]),
            #     nn.AvgPool1d(kernel_size=window_size[1]),
            #     nn.AvgPool1d(kernel_size=window_size[2])
            #     ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        # all_inputs.append(enc_input)
        all_inputs.append(enc_input.transpose(1, 2))

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            # all_inputs.append(enc_input)
            all_inputs.append(enc_input.transpose(1, 2))

        # all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        # all_inputs = self.norm(all_inputs)

        return all_inputs


class Predictor(nn.Module):

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        out = out
        return out


class Decoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt, mask):
        super().__init__()

        self.model_type = opt.model
        self.mask = mask

        self.layers = nn.ModuleList([
            DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                normalize_before=False),
            DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                normalize_before=False)
            ])

        if opt.embed_type == 'CustomEmbedding':
            self.dec_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.dec_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

    def forward(self, x_dec, x_mark_dec, refer):
        dec_enc = self.dec_embedding(x_dec, x_mark_dec)

        dec_enc, _ = self.layers[0](dec_enc, refer, refer)
        refer_enc = torch.cat([refer, dec_enc], dim=1)
        mask = self.mask.repeat(len(dec_enc), 1, 1).to(dec_enc.device)
        dec_enc, _ = self.layers[1](dec_enc, refer_enc, refer_enc, slf_attn_mask=mask)

        return dec_enc