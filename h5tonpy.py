#%%
import numpy as np
import tools_track_clean
import trackio
import contextlib
#%%
filename = "/scratch/hyun/RBC_challenge_data/fitted_conv_0_05ppp.h5"
for i in range(1,101):
    temp = tools_track_clean.read_general_trackio(
        filename,
        i,
        vars=["positions", "velocities", "accelerations", "pos_in_tracks", "tracklengths"],
        as_array=False,
        SI_Units=True,
        atts = None
    )
    data = np.concatenate([temp["positions"], temp["velocities"], temp["accelerations"]], axis=1)[32400:,:]
    with open("/scratch/hyun/RBC_challenge_data/fitted_0_05ppp_valid/ts_"+format(i, "04d")+".npy", "wb") as f:
        np.save(f, data)
    f.close()
# %%
print(np.max(data[:,0:3]))
# %%
print(data.shape)
# %%
print(data)
# %%
print(40500*0.8)
# %%
n = 40500/0.2**3
print(1/n**(1/3))
# %%
print(0.2/(1/n**(1/3)))
# %%
print(1.5314e-5/0.00342086)
# %%
