#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
#%%
checkpoints = ["TBL_SOAP_k1", "TBL_SOAP_k2", "TBL_SOAP_k4","TBL_SOAP_k8", "TBL_SOAP_k16", "TBL_SOAP_k32","TBL_SOAP_k64"]
t_error_list = []
for checkpoint in checkpoints:
    with open("datas/"+checkpoint+"/temporal_error.pkl", 'rb') as f:
        t_error = pickle.load(f)
    f.close()
    t_error_list.append(t_error)
loss_list = []
for checkpoint in checkpoints:
    with open("datas/"+checkpoint+"/error_evolution.pkl", 'rb') as f:
        t_error = pickle.load(f)
    f.close()
    loss_list.append(t_error)
acc_list = []
for checkpoint in checkpoints:
    with open("datas/"+checkpoint+"/acc_v.pkl", 'rb') as f:
        acc = pickle.load(f)
    f.close()
    acc_list.append(acc)

#%%
fig, axes = plt.subplots(4,7,figsize=(28,14))
title_list = ['downsampling k = 1', 'downsampling k = 2', 'downsampling k = 4', 'downsampling k = 8'
              , 'downsampling k = 16', 'downsampling k = 32', 'downsampling k = 64']
ylabel_list = ['velocity NRMSE', 'pressure NRMSE', 'acc NRMSE of training set', 'acc NRMSE of validation set']
for k in range(len(t_error_list)*t_error_list[0].shape[1]):
    axes[k%4,k//4].plot(t_error_list[k//4][:,k%4])
    if k%4 == 0:
        axes[k%4, k//4].set_title(title_list[k//4])
    if k//4 == 0:
        axes[k%4, k//4].set_ylabel(ylabel_list[k%4])
    axes[k%4,k//4].set_xlabel('timesteps')
plt.show()
#%%
plt.plot(acc[:100,1])
plt.plot(acc[:100,4])
plt.show()
# %%
