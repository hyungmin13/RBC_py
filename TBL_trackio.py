#%%
import contextlib
import trackio
import numpy as np

def get_times_in_trackio_file(infile):
    if "real_timesteps" in infile.handle:
        times = infile.handle["real_timesteps"].astype("int")
    elif "first_ts" in infile.metadata.attributes:
            dt = infile.metadata.attributes["delta_ts"]
            times = np.arange(infile.metadata.attributes["first_ts"], infile.metadata.attributes["last_ts"]+dt, dt)
    elif "times" in infile.metadata.attributes:
        times = infile.metadata.attributes["times"].astype("int") # NEW Code
    else:
        print(
            "WARNING: No times attached at groundTruthTimestepFile. Falling back to default"
        )
        times = range(0)

    return times

def read_general_trackio(
    filename,
    ts,
    vars=["positions", "velocities", "accelerations"],
    as_array=False,
    SI_Units=True,
    atts = None
):
    from scipy.spatial.transform import Rotation as R

    particles = []
    i = 3
    res = {"particles_found": False}
    f = filename
    with contextlib.closing(trackio.TimeStepFile.open(f, "r")) as infile:
        if atts == None:
            atts = infile.metadata.attributes
        if "velocities" in vars or "accelerations" in vars:
            assert "tScale" in atts and "direction" in atts
            if SI_Units:
                tScale = atts["tScale"]
                if "px_to_mm" in atts: # 3D measurements should operate in mm. This is typically only present for 2D measurements. Now obsolete. Use camera in 2D measurements!
                    vsize = atts["px_to_mm"]
                else:
                    vsize = 1.0
            else:
                tScale = 1.0 / atts["deltaT"]
                vsize = atts["vsize"]

        real_timesteps = get_times_in_trackio_file(infile)
        real_to_internal = dict(zip(real_timesteps, range(len(real_timesteps))))

        def get_index(real_timestep_index):
            if (real_timestep_index in real_to_internal) and (
                real_to_internal[real_timestep_index] <= len(infile) - 1
            ):
                return real_to_internal[real_timestep_index]
            else:
                return None

        # trackLengths = np.array(infile.handle["trackLengths"])
        # trackStartTimesteps = np.array(infile.handle["trackStartTimesteps"])
        timestepIndex = get_index(ts)
        if timestepIndex != None:
            timestep = infile[timestepIndex]
            tracklocations = infile.tracklocations(
                timestepIndex
            )  # yields trackId, tracklengths and pos_in_tracks

            if timestep != None:
                for v in vars:

                    if v == "pos_in_tracks":
                        val = tracklocations.pos_in_tracks
                    elif v == "tracklengths":
                        val = tracklocations.tracklengths
                    else:
                        val = eval("timestep.%s()" % v)
                        if v == "positions":
                            if "translation" in atts:
                                val += atts["translation"]
                            if "vibrations" in atts:
                                val -= atts["vibrations"][timestepIndex]
                            if "px_to_mm" in atts:
                                val *= atts["px_to_mm"]
                        if v == "velocities":
                            val *= tScale * atts["direction"] * vsize
                        if v == "accelerations":
                            val *= tScale * tScale * 1000 * vsize
                        if v in ["positions", "velocities", "accelerations"]:
                            if "axis_transform" in atts:
                                at = atts["axis_transform"]
                                val = np.column_stack(
                                    [
                                        np.sign(at[0]) * val[:, abs(at[0]) - 1],
                                        np.sign(at[1]) * val[:, abs(at[1]) - 1],
                                        np.sign(at[2]) * val[:, abs(at[2]) - 1],
                                    ]
                                )  # -1 necessary due to 0 not having a sign
                            if "rotation" in atts:
                                r = R.from_matrix(atts["rotation"])
                                val = r.apply(val)
                        #if v == "positions":
                        #    val[:,1]-=0.000035*np.power(val[:,0]-30.0, 2.0)
                            
                    res.update({v: val})
                if res["particles_found"] == False:
                    res["particles_found"] = True
        else:
            print("Timestep ", ts, " does not exist in data file!")
    if as_array:
        if res["particles_found"] == True:
            for i, v in enumerate(vars):
                if i == 0:
                    dats = res[v]
                else:
                    dats = np.column_stack([dats, res[v]])
            return dats
        else:
            return []
    else:
        return res
#%%
filename = "/home/bussard/hyun_sh/TBL_PINN/data/tracks_noise_00_fitted_co_0.3.trackio_ts"
ts = 10

dats = read_general_trackio(
    filename,
    ts,
    vars=["positions", "velocities", "accelerations","tracklengths"],
    as_array=False,
    SI_Units=True,
    atts = None
)
#%%
index = np.where(dats['tracklengths']==51)
#%%
print(dats['velocities'][index].shape)
#%%
import numpy as np
import scipy
#%%
a = scipy.io.loadmat('/home/bussard/hyun_sh/TBL_PINN/data/HIT/IsoturbFlow.mat')
# %%
import h5py
f = h5py.File('/home/bussard/hyun_sh/TBL_PINN/data/HIT/IsoturbFlow.mat','r')
# %%
print(f)
# %%
keys = f.keys()
# %%
print(keys)
# %%
u = f.get('u')
print(u)
# %%
u = np.array(u)
# %%
import matplotlib.pyplot as plt
# %%
plt.imshow(u[10,10,:,:],cmap='jet')
plt.show()
# %%
