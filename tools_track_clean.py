#%%
import sys
import os
import pathlib
import contextlib
import numpy as np
import json
import trackio



def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


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
            print("Timestep does not exist in data file!")
    if as_array:
        for i, v in enumerate(vars):
            if i == 0:
                dats = res[v]
            else:
                dats = np.column_stack([dats, res[v]])
        return dats
    else:
        return res


def exportFittedTracksToTecplot_trackio_single(
    filename, tecplot_file, ti, do_plt, SI_Units
):
    dats = read_general_trackio(
        filename,
        ti,
        [
            "positions",
            "velocities",
            "accelerations",
            "intensities",
            "track_ids",
            "tracklengths",
            "pos_in_tracks",
        ],
        as_array=False,
        SI_Units=SI_Units,
    )
    # print(dats)
    if dats["particles_found"] == False:
        print("No particle in timestep " + str(ti))
    else:
        if do_plt == False:
            formatstring = "%+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.3f %+5d %+5d %+5d %+5d"
            if SI_Units:
                np.savetxt(
                    tecplot_file,
                    dats,
                    fmt=formatstring,
                    comments="",
                    header='TITLE = "Particles_step32_substep0_fit11"\nVARIABLES = "x [mm]", "y [mm]", "z [mm]", "u [m/s]", "v [m/s]", "w [m/s]", "ax [m/s^2]", "ay [m/s^2]", "az [m/s^2]", "i [au]",  "time step",  "track-length",  "pos_in_track",  "track-id", "x_raw [mm]", "y_raw [mm]", "z_raw [mm]"',
                )
            else:
                np.savetxt(
                    tecplot_file,
                    dats,
                    fmt=formatstring,
                    comments="",
                    header='TITLE = "Particles_step32_substep0_fit11"\nVARIABLES = "x [mm]", "y [mm]", "z [mm]", "u [px/ts]", "v [px/ts]", "w [px/ts]", "ax [px^2/ts^2]", "ay [px/ts^2]", "az [px/ts^2]", "i [au]",  "time step",  "track-length",  "pos_in_track",  "track-id", "x_raw [mm]", "y_raw [mm]", "z_raw [mm]"',
                )
            print("Path: " + os.path.join(tecplot_folder, "ts_%06d.dat" % ti))
        else:
            import tecplot as tp

            pltfile = tecplot_file
            print(pltfile)
            # Neues Layout
            try:
                tp.new_layout()
                # Datensatz erzeugen
                if SI_Units:
                    # ds = tp.active_frame().create_dataset('Run_ID', ['x [mm]', 'y [mm]', 'z [mm]', 'u [m/s]', 'v [m/s]', 'w [m/s]', 'ax [m/s^2]', 'ay [m/s^2]', 'az [m/s^2]', 'I', 'time_step', 'track_length', 'pos_in_track', 'track_id'])
                    ds = tp.active_frame().create_dataset(
                        "Run_ID",
                        [
                            "x [mm]",
                            "y [mm]",
                            "z [mm]",
                            "u [m/s]",
                            "v [m/s]",
                            "w [m/s]",
                            "ax [m/s^2]",
                            "ay [m/s^2]",
                            "az [m/s^2]",
                            "I",
                            "track_length",
                            "pos_in_track",
                            "track_id",
                        ],
                    )
                    newzone = ds.add_zone(5, "t = %05d" % ti, len(dats["velocities"]))
                    newzone.values("u [[]m/s[]]")[:] = dats["velocities"][:, 0]
                    newzone.values("v [[]m/s[]]")[:] = dats["velocities"][:, 1]
                    newzone.values("w [[]m/s[]]")[:] = dats["velocities"][:, 2]
                    newzone.values("ax [[]m/s^2[]]")[:] = dats["accelerations"][:, 0]
                    newzone.values("ay [[]m/s^2[]]")[:] = dats["accelerations"][:, 1]
                    newzone.values("az [[]m/s^2[]]")[:] = dats["accelerations"][:, 2]
                else:
                    ds = tp.active_frame().create_dataset(
                        "Run_ID",
                        [
                            "x [mm]",
                            "y [mm]",
                            "z [mm]",
                            "u [px/ts]",
                            "v [px/ts]",
                            "w [px/ts]",
                            "ax [px/ts^2]",
                            "ay [px/ts^2]",
                            "az [px/ts^2]",
                            "I",
                            "track_length",
                            "pos_in_track",
                            "track_id",
                        ],
                    )
                    newzone = ds.add_zone(5, "t = %05d" % ti, len(dats["velocities"]))
                    newzone.values("u [[]px/ts[]]")[:] = dats["velocities"][:, 0]
                    newzone.values("v [[]px/ts[]]")[:] = dats["velocities"][:, 1]
                    newzone.values("w [[]px/ts[]]")[:] = dats["velocities"][:, 2]
                    newzone.values("ax [[]px/ts^2[]]")[:] = dats["accelerations"][:, 0]
                    newzone.values("ay [[]px/ts^2[]]")[:] = dats["accelerations"][:, 1]
                    newzone.values("az [[]px/ts^2[]]")[:] = dats["accelerations"][:, 2]

                newzone.values("x [[]mm[]]")[:] = dats["positions"][:, 0]
                newzone.values("y [[]mm[]]")[:] = dats["positions"][:, 1]
                newzone.values("z [[]mm[]]")[:] = dats["positions"][:, 2]
                newzone.values("I")[:] = dats["intensities"]
                # newzone.values('time_step')[:] = dats['time_steps']
                newzone.values("track_length")[:] = dats["tracklengths"]
                newzone.values("pos_in_track")[:] = dats["pos_in_tracks"]
                newzone.values("track_id")[:] = dats["track_ids"]
                # newzone.values('x_raw [[]mm[]]')[:] = dats[:,14]
                # newzone.values('y_raw [[]mm[]]')[:] = dats[:,15]
                # newzone.values('z_raw [[]mm[]]')[:] = dats[:,16]
                tp.data.save_tecplot_plt(pltfile, dataset=ds)
            except:
                print('Exception in exportFittedTracksToTecplot_trackio_single: Tecplot Licence error?')



def exportRawTracksToTecplot_trackio_thread(inputs):
    (filename, tecplot_folder, time_steps, do_plt, atts) = inputs
    for ti in time_steps:
        dats = read_general_trackio(
            filename,
            ti,
            [
                "positions",
                "intensities",
                "track_ids",
                "tracklengths",
                "pos_in_tracks",
            ],
            as_array=False,
            SI_Units=False,
            atts = atts,
        )
        # print(dats)
        if dats["particles_found"] == False:
            print("No particle in timestep " + str(ti))
        else:
            if do_plt == False:
                formatstring = "%+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.3f %+5d %+5d %+5d %+5d"
                fn = "ts_%05d.dat" % (ti)
                print(fn)
                if SI_Units:
                    np.savetxt(
                        os.path.join(tecplot_folder, fn),
                        dats,
                        fmt=formatstring,
                        comments="",
                        header='TITLE = "Particles_step32_substep0_fit11"\nVARIABLES = "x [mm]", "y [mm]", "z [mm]", "u [m/s]", "v [m/s]", "w [m/s]", "ax [m/s^2]", "ay [m/s^2]", "az [m/s^2]", "i [au]",  "time step",  "track-length",  "pos_in_track",  "track-id", "x_raw [mm]", "y_raw [mm]", "z_raw [mm]"',
                    )
                else:
                    np.savetxt(
                        os.path.join(tecplot_folder, fn),
                        dats,
                        fmt=formatstring,
                        comments="",
                        header='TITLE = "Particles_step32_substep0_fit11"\nVARIABLES = "x [mm]", "y [mm]", "z [mm]", "u [px/ts]", "v [px/ts]", "w [px/ts]", "ax [px^2/ts^2]", "ay [px/ts^2]", "az [px/ts^2]", "i [au]",  "time step",  "track-length",  "pos_in_track",  "track-id", "x_raw [mm]", "y_raw [mm]", "z_raw [mm]"',
                    )
                print("Path: " + os.path.join(tecplot_folder, "ts_%06d.dat" % ti))
            else:
                import tecplot as tp

                if os.path.isdir(tecplot_folder):
                    pltfile = os.path.join(tecplot_folder, "ts_%05d.plt" % (ti))
                elif os.path.isfile(tecplot_folder):
                    pltfile = tecplot_folder
                # print(pltfile)
                # Neues Layout
                try:

                    tp.new_layout()
                    # Datensatz erzeugen
                    ds = tp.active_frame().create_dataset(
                            "Run_ID",
                            [
                                "x [mm]",
                                "y [mm]",
                                "z [mm]",
                                "I",
                                "track_length",
                                "pos_in_track",
                                "track_id",
                            ],
                        )
                    newzone = ds.add_zone(5, "t = %05d" % ti, len(dats["positions"]))
                    newzone.values("x [[]mm[]]")[:] = dats["positions"][:, 0]
                    newzone.values("y [[]mm[]]")[:] = dats["positions"][:, 1]
                    newzone.values("z [[]mm[]]")[:] = dats["positions"][:, 2]
                    newzone.values("I")[:] = dats["intensities"]
                    # newzone.values('time_step')[:] = dats['time_steps']
                    newzone.values("track_length")[:] = dats["tracklengths"]
                    newzone.values("pos_in_track")[:] = dats["pos_in_tracks"]
                    newzone.values("track_id")[:] = dats["track_ids"]
                    tp.data.save_tecplot_plt(pltfile, dataset=ds)
                except:
                    print('Exception in exportRawTracksToTecplot_trackio_thread: Tecplot Licence error?')


def exportFittedTracksToTecplot_trackio_thread(inputs):
    (filename, tecplot_folder, time_steps, do_plt, SI_Units, atts) = inputs
    for ti in time_steps:
        dats = read_general_trackio(
            filename,
            ti,
            [
                "positions",
                "velocities",
                "accelerations",
                "intensities",
                "track_ids",
                "tracklengths",
                "pos_in_tracks",
            ],
            as_array=False,
            SI_Units=SI_Units,
            atts = atts,
        )
        # print(dats)
        if dats["particles_found"] == False:
            print("No particle in timestep " + str(ti))
        else:
            if do_plt == False:
                formatstring = "%+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f %+9.3f %+5d %+5d %+5d %+5d"
                fn = "ts_%05d.dat" % (ti)
                print(fn)
                if SI_Units:
                    np.savetxt(
                        os.path.join(tecplot_folder, fn),
                        dats,
                        fmt=formatstring,
                        comments="",
                        header='TITLE = "Particles_step32_substep0_fit11"\nVARIABLES = "x [mm]", "y [mm]", "z [mm]", "u [m/s]", "v [m/s]", "w [m/s]", "ax [m/s^2]", "ay [m/s^2]", "az [m/s^2]", "i [au]",  "time step",  "track-length",  "pos_in_track",  "track-id", "x_raw [mm]", "y_raw [mm]", "z_raw [mm]"',
                    )
                else:
                    np.savetxt(
                        os.path.join(tecplot_folder, fn),
                        dats,
                        fmt=formatstring,
                        comments="",
                        header='TITLE = "Particles_step32_substep0_fit11"\nVARIABLES = "x [mm]", "y [mm]", "z [mm]", "u [px/ts]", "v [px/ts]", "w [px/ts]", "ax [px^2/ts^2]", "ay [px/ts^2]", "az [px/ts^2]", "i [au]",  "time step",  "track-length",  "pos_in_track",  "track-id", "x_raw [mm]", "y_raw [mm]", "z_raw [mm]"',
                    )
                print("Path: " + os.path.join(tecplot_folder, "ts_%06d.dat" % ti))
            else:
                import tecplot as tp

                if os.path.isdir(tecplot_folder):
                    pltfile = os.path.join(tecplot_folder, "ts_%05d.plt" % (ti))
                elif os.path.isfile(tecplot_folder):
                    pltfile = tecplot_folder
                # print(pltfile)
                # Neues Layout
                try:
                    tp.new_layout()
                    # Datensatz erzeugen
                    if SI_Units:
                        # ds = tp.active_frame().create_dataset('Run_ID', ['x [mm]', 'y [mm]', 'z [mm]', 'u [m/s]', 'v [m/s]', 'w [m/s]', 'ax [m/s^2]', 'ay [m/s^2]', 'az [m/s^2]', 'I', 'time_step', 'track_length', 'pos_in_track', 'track_id'])
                        ds = tp.active_frame().create_dataset(
                            "Run_ID",
                            [
                                "x [mm]",
                                "y [mm]",
                                "z [mm]",
                                "u [m/s]",
                                "v [m/s]",
                                "w [m/s]",
                                "ax [m/s^2]",
                                "ay [m/s^2]",
                                "az [m/s^2]",
                                "I",
                                "track_length",
                                "pos_in_track",
                                "track_id",
                            ],
                        )
                        newzone = ds.add_zone(5, "t = %05d" % ti, len(dats["velocities"]))
                        newzone.values("u [[]m/s[]]")[:] = dats["velocities"][:, 0]
                        newzone.values("v [[]m/s[]]")[:] = dats["velocities"][:, 1]
                        newzone.values("w [[]m/s[]]")[:] = dats["velocities"][:, 2]
                        newzone.values("ax [[]m/s^2[]]")[:] = dats["accelerations"][:, 0]
                        newzone.values("ay [[]m/s^2[]]")[:] = dats["accelerations"][:, 1]
                        newzone.values("az [[]m/s^2[]]")[:] = dats["accelerations"][:, 2]
                    else:
                        ds = tp.active_frame().create_dataset(
                            "Run_ID",
                            [
                                "x [mm]",
                                "y [mm]",
                                "z [mm]",
                                "u [px/ts]",
                                "v [px/ts]",
                                "w [px/ts]",
                                "ax [px/ts^2]",
                                "ay [px/ts^2]",
                                "az [px/ts^2]",
                                "I",
                                "track_length",
                                "pos_in_track",
                                "track_id",
                            ],
                        )
                        newzone = ds.add_zone(5, "t = %05d" % ti, len(dats["velocities"]))
                        newzone.values("u [[]px/ts[]]")[:] = dats["velocities"][:, 0]
                        newzone.values("v [[]px/ts[]]")[:] = dats["velocities"][:, 1]
                        newzone.values("w [[]px/ts[]]")[:] = dats["velocities"][:, 2]
                        newzone.values("ax [[]px/ts^2[]]")[:] = dats["accelerations"][:, 0]
                        newzone.values("ay [[]px/ts^2[]]")[:] = dats["accelerations"][:, 1]
                        newzone.values("az [[]px/ts^2[]]")[:] = dats["accelerations"][:, 2]

                    newzone.values("x [[]mm[]]")[:] = dats["positions"][:, 0]
                    newzone.values("y [[]mm[]]")[:] = dats["positions"][:, 1]
                    newzone.values("z [[]mm[]]")[:] = dats["positions"][:, 2]
                    newzone.values("I")[:] = dats["intensities"]
                    # newzone.values('time_step')[:] = dats['time_steps']
                    newzone.values("track_length")[:] = dats["tracklengths"]
                    newzone.values("pos_in_track")[:] = dats["pos_in_tracks"]
                    newzone.values("track_id")[:] = dats["track_ids"]
                    # newzone.values('x_raw [[]mm[]]')[:] = dats[:,14]
                    # newzone.values('y_raw [[]mm[]]')[:] = dats[:,15]
                    # newzone.values('z_raw [[]mm[]]')[:] = dats[:,16]
                    tp.data.save_tecplot_plt(pltfile, dataset=ds)
                except:
                    print('Exception in exportFittedTracksToTecplot_trackio_thread: Tecplot Licence error?')


def exportFittedTracksToTecplot_trackio_parallel(
    h5_folder, out_folder, times, numThreads=16, do_plt=True, SI_Units=True, atts = None, mode = 'fitted'
):

    import multiprocessing

    print("opening hdf5folder: " + h5_folder)
    if numThreads < 1:
        print("Please provide a number of threads >= 1")
        return

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if numThreads > 1:
        with contextlib.closing(multiprocessing.Pool(numThreads)) as pool:
            ts_segments = list(split(times, numThreads))
            print("ts_segments: ", ts_segments)
            if mode == 'fitted':
                pool.map(
                    exportFittedTracksToTecplot_trackio_thread,
                    (
                        (h5_folder, out_folder, ts_segments[t], do_plt, SI_Units, atts)
                        for t in range(numThreads)
                    )
                )
            else:
                pool.map(
                    exportRawTracksToTecplot_trackio_thread,
                    (
                        (h5_folder, out_folder, ts_segments[t], do_plt, atts)
                        for t in range(numThreads)
                    )
                )
    else:
        if mode == 'fitted':
            exportFittedTracksToTecplot_trackio_thread(
                (h5_folder, out_folder, times, do_plt, SI_Units, atts))
        else:
            exportRawTracksToTecplot_trackio_thread(
                (h5_folder, out_folder, times, do_plt, atts))



def exportIPRParticlesToTecplot(
    ipr, tecplot_path, timeName, do_plt
):
    particles = ipr.getParticles().as_array()
    dats = {"positions": particles[:,0:3], "intensities": particles[:,3]}
    # print(dats)
    if do_plt == False:
        tecplot_file = str(tecplot_path.joinpath('IPR_Particles_ts_' + timeName + '.dat'))
        formatstring = "%+9.4f %+9.4f %+9.4f %+9.4f"
        np.savetxt(
            tecplot_file,
            dats,
            fmt=formatstring,
            comments="",
            header='TITLE = "Particles_step32_substep0_fit11"\nVARIABLES = "x [mm]", "y [mm]", "z [mm]", "i [au]"',
        )
        print("Path for tecplot particle export: " + tecplot_file)
    else:
        import tecplot as tp
        tecplot_file = str(tecplot_path.joinpath('IPR_Particles_ts_' + timeName + '.plt'))
        print(tecplot_file)
        # Neues Layout
        try:
            tp.new_layout()
            # Datensatz erzeugen
            ds = tp.active_frame().create_dataset(
                "Run_ID",
                [
                    "x [mm]",
                    "y [mm]",
                    "z [mm]",
                    "I",
                ],
            )
            newzone = ds.add_zone(5, "t = %s" % timeName, len(dats["positions"]))
            
            newzone.values("x [[]mm[]]")[:] = dats["positions"][:, 0]
            newzone.values("y [[]mm[]]")[:] = dats["positions"][:, 1]
            newzone.values("z [[]mm[]]")[:] = dats["positions"][:, 2]
            newzone.values("I")[:] = dats["intensities"]
            # newzone.values('time_step')[:] = dats['time_steps']
            print("Path for tecplot particle export: " + tecplot_file)
            tp.data.save_tecplot_plt(tecplot_file, dataset=ds)
        except:
            print('Exception in exportIPRParticlesToTecplot: Tecplot Licence error?')

def get_times_in_trackio_file(infile):
    print(infile.handle)
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

def convert_timestepFile_to_TrackFile(path):
    if type(path == str):
        path = pathlib.Path(path)
    if type(path == pathlib.PosixPath):
        decompress_call = "h5repack -v -l CONTI " + str(path) + " " + str(path)+".decompressed"
        convert_call = "ts_to_track " + str(path)+".decompressed "  + str(path)+".trackfile"
        print('decompressing: ', decompress_call)
        os.system(decompress_call)
        print('converting :', convert_call)
        os.system(convert_call)
        print('deleting decompressed file')
        os.system("rm " + str(path)+ ".decompressed ")
    else:
        print('Either a string or a pathlib.PosixPath object is needed')
