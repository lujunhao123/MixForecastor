import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
file = "Ramp_30min_24_1_MiTSformer_Pcgrad_30min_ftM_sl24_ll12_pl1_dm256_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
pred_con = np.load(r"/home/lx/lu/Ramp/ramp_Su1/test_results/" + file + "/pred_con.npy")
pred_dis = np.load(r"/home/lx/lu/Ramp/ramp_Su1/test_results/" + file + "/pred_dis.npy")
true_con = np.load(r"/home/lx/lu/Ramp/ramp_Su1/test_results/" + file + "/true_con.npy")
true_dis = np.load(r"/home/lx/lu/Ramp/ramp_Su1/test_results/" + file + "/true_dis.npy")
print(pred_con.shape, pred_dis.shape, true_con.shape, true_dis.shape)
pred_dis += 1
plt.figure(figsize=(10,5))
plt.plot(pred_con[:100,0,0], label="pred")
plt.plot(true_con[:100,0,0], label="true")

plt.legend()
plt.show()

import torch

plt.figure(figsize=(10,5))
pred_pro = torch.nn.functional.softmax(torch.tensor(pred_dis),dim=-1).numpy()
print(pred_pro.shape)
plt.plot(pred_pro[:100,0,0,0], label="pred1")
plt.plot(pred_pro[:100,0,0,1], label="pred2")
plt.plot(pred_pro[:100,0,0,2], label="pred3")
plt.plot(true_dis[:100,0,0], label="true")

plt.legend()
plt.show()