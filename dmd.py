"""
DMD Analysis Package
VERSION 1.0.0 // 05 SEP 17
Created on Mon Jun 27 2016
@author: Roy Taylor
University of Washington, Department of Aeronautics and Astronautics
"""

# CHANGE LOG
#
# 27 JUN 16 : RKT : Package created, not operational
# 09 AUG 16 : RKT : dmd function operational
# 10 AUG 16 : RKT : Added error handling, cleaned up, added mode_selector
# 11 AUG 16 : RKT : Added monotonicity. Release 1.0.0
# 05 SEP 17 : RKT : Total refactor for publication
#
# TODO add todos

import numpy as np
import warnings
from scipy.io import loadmat, savemat
from argparse import ArgumentParser


# Helper functions


def monotonic_increase(input_array):
    """
    Quickly determines if an array increases monotonically.
    :param input_array:
    :return:
    """
    dx = np.diff(input_array)
    return np.all(dx >= 0)


def load_from_mat(mat_file):
    """
    A method of limited intelligence for importing data matrix and
    time array from a .mat file.

    Function will search for entries labeled `frames` and `t`.
    Failing to find this, it will assume that the file contains these
    but labeled differently, and tries to infer from array sizes which
    is which.

    :param mat_file:
    :return:
    """
    file_data = loadmat(mat_file)
    try:
        frames = file_data["frames"]
        t = file_data["t"]
    except KeyError:
        a, b = file_data.values()[0], file_data.values()[1]
        if a.shape[0] > b.shape[0]:
            frames, t = a, b
        elif b.shape[0] > a.shape[0]:
            frames, t = b, a
        elif a.shape[0] == b.shape[0]:
            if a.shape[1] > b.shape[1]:
                frames, t = a, b
            elif b.shape[1] > a.shape[1]:
                frames, t = b, a
            else:
                raise IOError("Invalid input file.")
        else:
            raise IOError("Invalid input file")
    t = np.squeeze(t)
    if len(t) not in frames.shape:
        raise DMDError("Timebase not of same dimension as frames.")
    return frames, t







# Classes


class DMDError(Exception):
    pass


# Kernel functions


def dmd(frames, time_array, runahead_timesteps=0, top_modes=0):
    """

    :param frames:
        <dtype = numpy.ndarray>
        Data frame, such that each column corresponds to a time snapshot
    :param time_array:
        <dtype = numpy.ndarray, OR list>
        Monotonically increasing time axis for input_matrix.
    :param runahead_timesteps:
        <dtype = int>
        Number of timesteps forward for extrapolation.
        If runahead_timesteps == 0, no prediction.
    :param top_modes:
        <dtype = int>
        Truncate to this number of modes, from the top.
        If top_modes == 0, no truncation.
    :return:
    """

    # check runtime conditions and ensure data typing if necessary.
    runahead_timesteps = int(runahead_timesteps)
    top_modes = int(top_modes)
    if not monotonic_increase(time_array):
        raise ValueError("Timeseries data must increase monotonically.")
    if len(set(np.diff(time_array))) > 1:
        warnings.warn("Time array not evenly spaced.", RuntimeWarning)  # this triggers from round-off error...

    # structure data
    X = frames[:, 0:-1]
    Y = frames[:, 1:]
    x0 = frames[:, 0]
    dt = np.mean(np.diff(time_array))

    # lengthen time_array by runahead_timesteps, if nonzero
    if runahead_timesteps > 0:
        add_t0 = float(time_array[-1] + dt)
        add_tf = float(time_array[-1] + (dt * runahead_timesteps))
        add_time = np.linspace(add_t0, add_tf, runahead_timesteps)
        time_array = np.append(time_array, add_time)

    # perform initial decomposition and truncate, if applicable
    [U1, S1, V1] = np.linalg.svd(X, full_matrices=False)
    S2 = np.diag(S1)  # this is just as annoying in MATLAB, too.
    if top_modes:
        U = U1[:, 0:top_modes]
        S = S2[0:top_modes, 0:top_modes]
        V = V1[0:top_modes, :]
    else:
        U, S, V = U1, S2, V1
    V = V.transpose()

    # the DMD supposes there exists a linear map A such that AX = Y
    # we use the SVD of X to approximate A.
    dS = np.diag(np.divide(1, np.diag(S)))
    vs = np.dot(V, dS)
    A = np.dot(U.transpose(), np.dot(Y, vs))

    # the eigenstates of a linear map describe its modes.
    # we compute these eigenstates to derive the DMD spectra
    [vals, vecs] = np.linalg.eig(A)
    omega = np.divide(np.log(vals), dt)

    # we re-represent the modes as a time-series
    phi = np.dot(U, vecs)
    mps = phi.shape[1]
    y_0 = np.linalg.lstsq(phi, x0)[0]
    mod = np.zeros(mps).reshape(mps, 1)
    for k in range(0, len(time_array)-2):
        mod_update = (y_0 * np.e ** (omega*time_array[k])).reshape(mps, 1)
        mod = np.append(mod, mod_update, axis=1)

    return phi, mod, omega, np.diag(S)


def sliding_window_dmd(frames, t, window_length, top_modes=0):
    """

    :param frames:
    :param t:
    :param window_length:
    :param top_modes:
    :return:
    """
    frequency, energy = [], []
    for k in range(0, len(t) - window_length):
        _, _, o, s = dmd(frames[:, k:(k+window_length)], t[k:(k+window_length)], top_modes=top_modes)
        frequency.append(o)
        energy.append(s)
    freqs = np.imag(np.array(frequency)) / (np.pi * 2.0)
    return freqs, np.array(energy)



def calc_stationary_mode(frequency_array, energy_array):
    """

    :param frequency_array:
    :param energy_array:
    :return:
    """


def mode_selector(phi, mod, mode=None):
    """

    :param phi:
    :param mod:
    :param mode:
    :return:
    """

    if mode is None:
        data = np.dot(phi, mod)
    else:
        phi_dim, mod_dim = phi.shape[0], mod.shape[1]
        mud = phi[:, mode].reshape(phi_dim, 1)
        kip = mod[mode, :].reshape(1, mod_dim)
        data = np.dot(mud, kip)
    return data







