from scipy.optimize import minimize
from scipy.optimize import nnls
import numpy as np
import torch

import random
import copy


# ----------------------------------------------------------------------------------------------------------------------
# Pareto(model)
# ----------------------------------------------------------------------------------------------------------------------

def ASM(hat_w, c):
    """
    ref:
    http://ofey.me/papers/Pareto.pdf,
    https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1
    :param hat_w: # (K,)
    :param c: # (K,)
    :return:
    """
    A = np.array([[0 if i != j else 1 for i in range(len(c))] for j in range(len(c))])
    b = hat_w
    x0, _ = nnls(A, b)

    def _fn(x, A, b):
        return np.linalg.norm(A.dot(x) - b)

    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) + np.sum(c) - 1}
    bounds = [[0., None] for _ in range(len(hat_w))]
    min_out = minimize(_fn, x0, args=(A, b), method='SLSQP', bounds=bounds, constraints=cons)
    new_w = min_out.x + c
    return new_w

def pareto_step(w, c, G):
    """
    ref:http://ofey.me/papers/Pareto.pdf
    K : the number of task
    M : the dim of NN's params
    :param W: # (K,1)
    :param C: # (K,1)
    :param G: # (K,M)
    :return:
    """
    GGT = np.matmul(G, np.transpose(G))  # (K, K)
    e = np.mat(np.ones(np.shape(w)))  # (K, 1)
    m_up = np.hstack((GGT, e))  # (K, K+1)
    m_down = np.hstack((np.transpose(e), np.mat(np.zeros((1, 1)))))  # (1, K+1)
    M = np.vstack((m_up, m_down))  # (K+1, K+1)
    z = np.vstack((-np.matmul(GGT, c), 1 - np.sum(c)))  # (K+1, 1)
    hat_w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(M), M)), M), z)  # (K+1, 1)
    hat_w = hat_w[:-1]  # (K, 1)
    hat_w = np.reshape(np.array(hat_w), (hat_w.shape[0],))  # (K,)
    c = np.reshape(np.array(c), (c.shape[0],))  # (K,)
    new_w = ASM(hat_w, c)
    return new_w

def apply_gradient(model,loss):
    model.zero_grad()
    loss.backward(retain_graph=True)

def pareto_fn(w_list, c_list, model,loss_list,number_task):
    grads = [[] for i in range(len(loss_list))]

    for i,loss in enumerate(loss_list):
        for p in model.parameters():
            if p.grad is not None:
                grads[i].append(p.grad.view(-1))
            else:
                grads[i].append(torch.zeros_like(p).cuda(non_blocking=True).view(-1))

        grads[i] = torch.cat(grads[i],dim=-1).cpu().numpy()



    grads = np.concatenate(grads,axis=0).reshape(number_task,-1)
    weights = np.mat([[w] for w in w_list])
    c_mat = np.mat([[c] for c in c_list])
    new_w_list = pareto_step(weights, c_mat, grads)

    return new_w_list


# ----------------------------------------------------------------------------------------------------------------------
# PCGrad(model)
# ----------------------------------------------------------------------------------------------------------------------


def get_gradient(model, loss):
    model.zero_grad()

    loss.backward(retain_graph=True)



def set_gradient(grads, optimizer, shapes):
    for group in optimizer.param_groups:
        length = 0
        for i, p in enumerate(group['params']):
            # if p.grad is None: continue
            i_size = np.prod(shapes[i])
            get_grad = grads[length:length + i_size]
            length += i_size
            p.grad = get_grad.view(shapes[i])


def pcgrad_fn(model, losses, optimizer, mode='mean'):
    grad_list = []
    shapes = []
    shares = []
    for i, loss in enumerate(losses):
        get_gradient(model, loss)
        grads = []
        for p in model.parameters():
            if i == 0:
                shapes.append(p.shape)
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros_like(p).view(-1))
        new_grad = torch.cat(grads, dim=0)
        grad_list.append(new_grad)

        if shares == []:
            shares = (new_grad != 0)
        else:
            shares &= (new_grad != 0)
    #clear memory
    loss_all = 0
    for los in losses:
        loss_all += los
    loss_all.backward()
    grad_list2 = copy.deepcopy(grad_list)
    for g_i in grad_list:
        random.shuffle(grad_list2)
        for g_j in grad_list2:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)

    grads = torch.cat(grad_list, dim=0)
    grads = grads.view(len(losses), -1)
    if mode == 'mean':
        grads_share = grads * shares.float()

        grads_share = grads_share.mean(dim=0)
        grads_no_share = grads * (1 - shares.float())
        grads_no_share = grads_no_share.sum(dim=0)

        grads = grads_share + grads_no_share
    else:
        grads = grads.sum(dim=0)

    set_gradient(grads, optimizer, shapes)
    return loss_all
