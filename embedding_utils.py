# -*- coding: utf-8 -*-
"""
Embedding and behavioral analysis utilities.

This module provides a collection of utility functions and classes for:
- Time-delay embedding and basis transformations
- UMAP and parametric UMAP embeddings
- Kernel density estimation and watershed-based spatial clustering
- Poselet extraction and segmentation
- Signal processing utilities (wavelets, gradients, angular metrics)
- Visualization and animation helpers

Originally created on Mon Jan 25 15:10:07 2021.

@author: ravindrannai
"""

import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pywt
import umap
from KDEpy import FFTKDE
from skimage.feature import peak_local_max
from scipy import ndimage, signal
from skimage.segmentation import watershed
from matplotlib.animation import FuncAnimation
import shelve
from itertools import groupby
from glm_utils import bases, preprocessing
from umap.parametric_umap import ParametricUMAP
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import time
import matplotlib as mpl
from joblib import Parallel, delayed
import tensorflow as tf


# =============================================================================
# Utility functions
# =============================================================================

def save_figure_data(fig, data, folder, figname, dpi=300, figfmt=".svg",
                     pickle_data=False, rewrite=True):
    """
    Save a matplotlib figure along with associated data to disk.

    Args:
        fig : matplotlib.figure.Figure
            Figure object to save.
        data : dict or pandas.DataFrame or None
            Data associated with the figure.
        folder : str
            Output directory.
        figname : str
            Base filename (without extension).
        dpi : int, optional
            Figure resolution.
        figfmt : str, optional
            Figure file format.
        pickle_data : bool, optional
            Whether to pickle dictionary data instead of saving as NPZ.
        rewrite : bool, optional
            Whether to overwrite existing files.
    """
    if rewrite or not os.path.exists(os.path.join(folder, figname + figfmt)):
        fig.savefig(os.path.join(folder, figname + figfmt), dpi=dpi)

    if data is not None:
        if isinstance(data, dict):
            if not pickle_data:
                if rewrite or not os.path.exists(os.path.join(folder, figname + ".npz")):
                    np.savez(os.path.join(folder, figname + ".npz"), **data)
            else:
                if rewrite or not os.path.exists(os.path.join(folder, figname + ".pickle")):
                    with open(os.path.join(folder, figname + ".pickle"), "wb") as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        elif isinstance(data, pd.DataFrame):
            if rewrite or not os.path.exists(os.path.join(folder, figname + ".csv")):
                data.to_csv(os.path.join(folder, figname + ".csv"))


def time_delay_embedding(X_data, y_data=None, nb_delays=50,
                         indices=None, multi_features=False, padding=None,
                         remove_nans=False):
    """
    Compute a time-delay embedding of input features.

    Args:
        X_data : ndarray
            Input feature matrix (time x features).
        y_data : ndarray or None
            Optional target values.
        nb_delays : int
            Number of delay steps.
        indices : array-like or None
            Indices to include.
        multi_features : bool
            Whether to flatten features inside the delay window.
        padding : {'same', None}
            Pad input to preserve original length.
        remove_nans : bool
            Remove rows containing NaNs.

    Returns:
        Xs : ndarray
            Time-delay embedded features.
        ys : ndarray, optional
            Corresponding targets.
    """
    nb_points, nb_stim = X_data.shape

    if padding == "same":
        X_data = np.pad(X_data, ((nb_delays, 0), (0, 0)))
        if y_data is not None:
            y_data = np.pad(y_data, (nb_delays, 0))

    Xs_ys = preprocessing.time_delay_embedding(
        X_data,
        y_data,
        window_size=nb_delays,
        flatten_inside_window=multi_features,
        indices=indices
    )

    Xs = Xs_ys if y_data is None else Xs_ys[0]
    nan_rows = np.isnan(Xs).any(1) if remove_nans else np.zeros(len(Xs), bool)

    if y_data is None:
        return Xs[~nan_rows]
    else:
        Xs, ys = Xs_ys
        return Xs[~nan_rows], ys[~nan_rows]


def basis_transformation(Xs, nb_delays=50, basis_fn=None,
                         multi_features=False, nb_stim=None):
    """
    Project time-delay embedded features onto a temporal basis.

    Args:
        Xs : ndarray
            Time-delay embedded features.
        nb_delays : int
            Number of delays.
        basis_fn : ndarray or None
            Custom basis functions.
        multi_features : bool
            Whether multiple features are flattened.
        nb_stim : int or None
            Number of stimulus dimensions.

    Returns:
        Xs_b : ndarray
            Basis-transformed features.
        basis_projection : preprocessing.BasisProjection
            Projection object.
    """
    if basis_fn is None:
        B = bases.raised_cosine(1, 7, [0, 12], 1, w=nb_delays)
    else:
        B = basis_fn

    if not multi_features:
        nb_points, nb_delays, nb_stim = Xs.shape
        basis_projection = preprocessing.BasisProjection(B)
        _, nbases = B.shape
        Xs_b = np.empty((nb_points, nbases, nb_stim))
        for stim in range(nb_stim):
            Xs_b[:, :, stim] = basis_projection.transform(Xs[:, :, stim])
    else:
        B_multi = bases.multifeature_basis(B, nb_stim)
        basis_projection = preprocessing.BasisProjection(B_multi)
        Xs_b = basis_projection.transform(Xs)

    return Xs_b, basis_projection


def get_state_dwell_times(state_sequence):
    """
    Compute dwell times for each discrete state.

    Args:
        state_sequence : array-like
            Sequence of discrete states.

    Returns:
        dict
            Mapping from state to list of dwell durations.
    """
    dwell_times = {state: [] for state in np.unique(state_sequence)}
    grouped_seq_lengths = [(k, sum(1 for _ in g)) for k, g in groupby(state_sequence)]
    for state, dwell_time in grouped_seq_lengths:
        dwell_times[state].append(dwell_time)
    return dwell_times


def get_state_transition_indices(state_sequence, transition_of_interest,
                                 min_recurrent_duration_samples=1):
    """
    Identify indices where specific state transitions occur.

    Args:
        state_sequence : array-like
            Discrete state sequence.
        transition_of_interest : tuple
            (source_states, destination_states).
        min_recurrent_duration_samples : int
            Minimum dwell time before and after transition.

    Returns:
        ndarray
            Indices of detected transitions.
    """
    if min_recurrent_duration_samples < 1:
        min_recurrent_duration_samples = 1

    is_transition = np.zeros_like(state_sequence, bool)
    unique_states = np.unique(state_sequence)
    source_states, destination_states = transition_of_interest

    if source_states is None:
        source_states = [s for s in unique_states if s not in destination_states]
    if destination_states is None:
        destination_states = [s for s in unique_states if s not in source_states]

    source_states = list(source_states) if isinstance(source_states, (list, tuple)) else [source_states]
    destination_states = list(destination_states) if isinstance(destination_states, (list, tuple)) else [destination_states]

    is_transition[
        min_recurrent_duration_samples:len(is_transition) - min_recurrent_duration_samples + 1
    ] = [
        (len(np.unique(state_sequence[i - min_recurrent_duration_samples:i])) == 1) and
        (state_sequence[i - 1] in source_states) and
        (len(np.unique(state_sequence[i:i + min_recurrent_duration_samples])) == 1) and
        (state_sequence[i] in destination_states)
        for i in range(min_recurrent_duration_samples,
                       len(is_transition) - min_recurrent_duration_samples + 1)
    ]

    return np.where(is_transition)[0]

def fill_gaps(data, max_gap_length=200, gap_value=0, fill_value=1):
    """
    Fill short gaps in a sequence with a specified value.

    Args:
        data (array-like): Input sequence.
        max_gap_length (int): Maximum gap length to fill.
        gap_value (int or float): Value defining a gap.
        fill_value (int, float, or None): Value to fill gaps with.
            If None, uses the previous value.

    Returns:
        numpy.ndarray: Gap-corrected sequence.
    """
    grouped_data = [(k, sum(1 for _ in g)) for k, g in groupby(data)]
    prev_value, idx = grouped_data[0]
    corrected_data = np.array(data)

    for value, length in grouped_data[1:]:
        if value == gap_value and length < max_gap_length:
            corrected_data[idx:idx + length] = prev_value if fill_value is None else fill_value
        else:
            prev_value = value
        idx += length

    return corrected_data


def remove_short_segments(data, min_segment_length, segment_label=None):
    """
    Remove short segments by replacing them with the preceding segment.

    Args:
        data (array-like): Input labeled sequence.
        min_segment_length (int): Minimum allowed segment length.
        segment_label (optional): Only remove segments with this label.

    Returns:
        numpy.ndarray: Corrected sequence.
    """
    grouped_data = [(k, sum(1 for _ in g)) for k, g in groupby(data)]
    prev_segment, idx = grouped_data[0]
    corrected_data = np.array(data)

    for segment, length in grouped_data[1:]:
        if segment_label is not None and segment != segment_label:
            prev_segment = segment
            idx += length
            continue

        if length < min_segment_length:
            corrected_data[idx:idx + length] = prev_segment
        else:
            prev_segment = segment
        idx += length

    return corrected_data


def unwrap_angle_metrics(angles, discont=180, fs=None, cutoff_freq=None, order=6):
    """
    Unwrap angular discontinuities and compute angular velocity and acceleration.

    Args:
        angles (numpy.ndarray): Angle time series in degrees.
        discont (float): Discontinuity threshold.
        fs (float, optional): Sampling frequency.
        cutoff_freq (float, optional): Low-pass filter cutoff.
        order (int): Filter order.

    Returns:
        tuple:
            - unwrapped_angles (numpy.ndarray)
            - angular_speed (numpy.ndarray)
            - angular_acceleration (numpy.ndarray)
    """
    if cutoff_freq is not None:
        Wn = cutoff_freq / (fs / 2) if fs is not None else cutoff_freq
        b, a = signal.butter(order, Wn, "low")

    angles_diff = np.diff(angles, axis=0)
    signchange_idx = np.where(np.abs(angles_diff) >= discont)[0]
    angles_unwrap = np.array(angles.ravel())

    for idx in signchange_idx:
        angles_unwrap[idx + 1:] -= angles_diff[idx]

    if cutoff_freq is not None:
        valid = ~np.isnan(angles_unwrap)
        angles_unwrap[valid] = signal.filtfilt(b, a, angles_unwrap[valid])

    angular_speed = np.gradient(angles_unwrap)
    angular_acceleration = np.gradient(angular_speed)

    return angles_unwrap, angular_speed, angular_acceleration


def get_confident_frames(poses_confidence, threshold):
    """
    Determine frames with sufficient pose confidence.

    Args:
        poses_confidence (numpy.ndarray): Confidence values per pose.
        threshold (float): Minimum confidence threshold.

    Returns:
        numpy.ndarray: Boolean mask of confident frames.
    """
    confident_frames = np.ones(poses_confidence.shape[:2], dtype=bool)
    if threshold is not None:
        confident_frames = np.min(poses_confidence, axis=2) >= threshold
    return confident_frames


def get_freq_scales(fmin, fmax, Nf, fs, spacing="uniform"):
    """
    Compute frequencies and corresponding wavelet scales.

    Args:
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        Nf (int): Number of frequencies.
        fs (float): Sampling rate.
        spacing (str): 'uniform' or 'dyadic'.

    Returns:
        tuple:
            - frequencies (numpy.ndarray)
            - scales (numpy.ndarray)
    """
    if spacing == "dyadic":
        freqs = np.array([
            fmax * 2 ** (-(i / (Nf - 1)) * np.log2(fmax / fmin))
            for i in range(Nf)
        ])
    else:
        freqs = np.linspace(fmin, fmax, Nf)

    scales = fs / freqs
    return freqs, scales


def normalize_pose_positions_fly(pose_positions_fly, poseparts, ref_poses=("head", "tail")):
    """
    Normalize egocentric pose positions by fly body length.

    Args:
        pose_positions_fly (numpy.ndarray): Pose positions.
        poseparts (array-like): Pose part names.
        ref_poses (tuple): Reference pose parts.

    Returns:
        numpy.ndarray: Normalized pose positions.
    """
    fly_length = np.nanmean(
        np.linalg.norm(
            pose_positions_fly[:, poseparts == ref_poses[0], :] -
            pose_positions_fly[:, poseparts == ref_poses[1], :],
            axis=1,
        )
    )
    return pose_positions_fly / fly_length

def compute_wavelet(time, signal, scales,
                    waveletname='cmor1.5-1.0',
                    plot=False,
                    cmap=plt.cm.seismic,
                    title='Wavelet Transform (Power Spectrum) of signal',
                    ylabel='Frequency (Hz)',
                    xlabel='Time'):
    """
    Compute the continuous wavelet transform and power spectrum of a signal.

    Args:
        time (numpy.ndarray): Time vector.
        signal (numpy.ndarray): Signal values.
        scales (numpy.ndarray): Wavelet scales.
        waveletname (str): Name of the wavelet.
        plot (bool): Whether to plot the result.
        cmap (matplotlib.colors.Colormap): Colormap for plotting.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        xlabel (str): X-axis label.

    Returns:
        tuple:
            - power (numpy.ndarray): Wavelet power spectrum.
            - time (numpy.ndarray): Time vector.
            - frequencies (numpy.ndarray): Corresponding frequencies.
    """
    dt = time[1] - time[0]
    signal = np.nan_to_num(signal)

    coefficients, frequencies = pywt.cwt(
        signal, scales, waveletname, dt, method='fft'
    )
    power = np.abs(coefficients) ** 2

    if plot and signal.ndim == 1:
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        contourlevels = np.log2(levels)
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))

        ax[0].plot(time, signal)
        ax[0].set_xlim(time[0], time[-1])

        im = ax[1].contourf(
            time,
            np.log2(frequencies),
            np.log2(power),
            contourlevels,
            extend='both',
            cmap=cmap,
        )

        ax[1].set_title(title)
        ax[1].set_ylabel(ylabel)
        ax[1].set_xlabel(xlabel)

        yticks = 2 ** np.arange(
            np.ceil(np.log2(frequencies.min())),
            np.ceil(np.log2(frequencies.max()))
        )
        ax[1].set_yticks(np.log2(yticks))
        ax[1].set_yticklabels(yticks)

        cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

    return power, time, frequencies


def get_poselets(pose_positions_xy, timestamps, poselet_length=10, stride_length=1):
    """
    Extract poselets (temporal windows of pose data).

    Args:
        pose_positions_xy (numpy.ndarray): Pose positions over time.
        timestamps (numpy.ndarray): Time vector.
        poselet_length (int): Length of each poselet.
        stride_length (int): Temporal stride.

    Returns:
        tuple:
            - poselets (numpy.ndarray)
            - poselets_poses (numpy.ndarray)
            - poselet_timestamps (list)
    """
    nb_poselets = int(pose_positions_xy.shape[0] / stride_length)

    pose_positions_padded = np.concatenate(
        (
            np.zeros((poselet_length // 2, pose_positions_xy.shape[1])),
            pose_positions_xy,
            np.zeros((poselet_length // 2, pose_positions_xy.shape[1])),
        ),
        axis=0,
    )

    timestamps_padded = np.concatenate(
        (np.zeros(poselet_length // 2), timestamps, np.zeros(poselet_length // 2))
    )

    poselets = [
        pose_positions_padded[p * stride_length:p * stride_length + poselet_length]
        for p in range(nb_poselets)
    ]

    poselet_timestamps = [
        timestamps_padded[p * stride_length + poselet_length // 2]
        for p in range(nb_poselets)
    ]

    poselets = np.stack(poselets)
    poselets_poses = np.nanmean(poselets, axis=1)

    return poselets, poselets_poses, poselet_timestamps


class UMAPEmbedding(umap.UMAP):
    """
    Extension of UMAP with reconstruction-based scoring.

    Provides prediction and scoring via inverse transform.
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1,
                 metric="euclidean", random_state=None):
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        self.fitted = False

    def fit(self, X, train_size=None):
        """
        Fit the UMAP model.

        Args:
            X (numpy.ndarray): Input data.
            train_size (float, optional): Fraction of data used for fitting.
        """
        if train_size is not None:
            train_idx = np.linspace(
                0, X.shape[0] - 1, int(X.shape[0] * train_size), dtype=int
            )
            X_train = X[train_idx]
        else:
            X_train = np.array(X)

        super().fit(X_train)
        self.fitted = True

    def predict(self, X):
        """
        Reconstruct input data from embedding.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Reconstructed data.
        """
        X_embedded = self.transform(X)
        return self.inverse_transform(X_embedded)

    def score(self, X_test, y_test=None, metric="rmse"):
        """
        Compute reconstruction error.

        Args:
            X_test (numpy.ndarray): Test data.
            y_test: Unused.
            metric (str): 'rmse' or 'mae'.

        Returns:
            float: Error score.
        """
        X_pred = self.predict(X_test)
        if metric == "rmse":
            return mean_squared_error(
                X_test.flatten(), X_pred.flatten(), squared=False
            )
        elif metric == "mae":
            return mean_absolute_error(X_test.flatten(), X_pred.flatten())
        else:
            raise NotImplementedError


class AutoEncoderUMAP(ParametricUMAP):
    """
    Parametric UMAP using an autoencoder architecture.
    """

    def __init__(
        self,
        input_dim,
        num_encoder_layers=(100, 100, 100),
        num_decoder_layers=(100, 100, 100),
        activation="relu",
        min_dist=0.1,
        n_neighbors=15,
        random_state=None,
        n_components=2,
        reconstruction_validation=None,
        parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
        verbose=True,
        n_training_epochs=1,
    ):
        """
        Initialize AutoEncoderUMAP.

        Args:
            input_dim (tuple): Input dimensionality.
            num_encoder_layers (tuple): Encoder layer sizes.
            num_decoder_layers (tuple): Decoder layer sizes.
            activation (str): Activation function.
            min_dist (float): UMAP minimum distance.
            n_neighbors (int): UMAP neighbors.
            random_state (int, optional): Random seed.
            n_components (int): Embedding dimensionality.
            reconstruction_validation (numpy.ndarray, optional): Validation set.
            parametric_reconstruction_loss_fcn: Reconstruction loss.
            verbose (bool): Verbosity.
            n_training_epochs (int): Training epochs.
        """
        self.input_dim = input_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.n_components = n_components
        self.activation = activation
        self.parametric_reconstruction_loss_fcn = parametric_reconstruction_loss_fcn
        self.verbose = verbose
        self.n_training_epochs = n_training_epochs
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

        encoder_layers = [tf.keras.layers.InputLayer(input_shape=input_dim[0])]
        for units in num_encoder_layers:
            encoder_layers.append(tf.keras.layers.Dense(units, activation=activation))
        encoder_layers.append(tf.keras.layers.Dense(n_components))
        self.encoder = tf.keras.Sequential(encoder_layers)

        decoder_layers = [tf.keras.layers.InputLayer(input_shape=n_components)]
        for units in num_decoder_layers:
            decoder_layers.append(tf.keras.layers.Dense(units, activation=activation))
        decoder_layers.append(tf.keras.layers.Dense(input_dim[0]))
        self.decoder = tf.keras.Sequential(decoder_layers)

        keras_fit_kwargs = {
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    min_delta=1e-2,
                    patience=10,
                    verbose=1,
                )
            ]
        }

        super().__init__(
            encoder=self.encoder,
            decoder=self.decoder,
            dims=self.input_dim,
            parametric_reconstruction=True,
            reconstruction_validation=reconstruction_validation,
            parametric_reconstruction_loss_fcn=self.parametric_reconstruction_loss_fcn,
            autoencoder_loss=True,
            verbose=self.verbose,
            keras_fit_kwargs=keras_fit_kwargs,
            n_training_epochs=self.n_training_epochs,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )

def umap_multiple_fits(
    X,
    model=UMAPEmbedding,
    n_neighbors_grid=[20, 50, 100, 200],
    min_dist_grid=[0.0, 0.1, 0.2, 0.5],
    metric_grid=["euclidean"],
    n_components_grid=[2],
    nb_delays_grid=[15, 30, 60, 120],
    train_size=0.1,
    val_size=0.1,
    random_state=42,
    visualize_results=True,
    verbose=True,
    n_jobs=None,
):
    """
    Perform grid search over UMAP hyperparameters with optional visualization.

    Args:
        X (array-like or list): Input data or list of trials.
        model (class): UMAP model class.
        n_neighbors_grid (list): Neighbor values.
        min_dist_grid (list): Minimum distance values.
        metric_grid (list): Distance metrics.
        n_components_grid (list): Embedding dimensionalities.
        nb_delays_grid (list): Delay embedding sizes.
        train_size (float): Training fraction.
        val_size (float): Validation fraction.
        random_state (int): Random seed.
        visualize_results (bool): Whether to plot embeddings.
        verbose (bool): Verbosity flag.
        n_jobs (int, optional): Parallel jobs.

    Returns:
        dict: Fit results indexed by hyperparameter combinations.
    """
    fit_results = {}

    for nb_delays in nb_delays_grid:
        if isinstance(X, list):
            X_delay = []
            for trial_X in X:
                X_td = time_delay_embedding(
                    trial_X, nb_delays=nb_delays, multi_features=True
                )
                X_delay.append(X_td)
            X_delay = np.concatenate(X_delay, axis=0)
        else:
            X_delay = time_delay_embedding(
                X, nb_delays=nb_delays, multi_features=True, remove_nans=True
            )

        train_indices = np.zeros(len(X_delay), bool)
        train_indices[np.arange(0, len(train_indices), int(1 / train_size))] = True
        X_train = X_delay[train_indices]

        if val_size is not None:
            _, X_val = train_test_split(
                X_delay[~train_indices], test_size=val_size
            )

        for n_components in n_components_grid:
            for n_neighbors in n_neighbors_grid:
                for min_dist in min_dist_grid:
                    for metric in metric_grid:
                        args = dict(
                            n_components=n_components,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric=metric,
                        )
                        if verbose:
                            print(args)

                        result = umap_train_validate(
                            X_train,
                            X_val,
                            model=model,
                            random_state=random_state,
                            **args,
                        )
                        fit_results[str(args)] = result

        if visualize_results:
            for metric in metric_grid:
                fig, ax = plt.subplots(
                    len(n_neighbors_grid),
                    len(min_dist_grid),
                    sharex=True,
                    sharey=True,
                    figsize=(3 * len(min_dist_grid), 3 * len(n_neighbors_grid)),
                    squeeze=False,
                )
                for i, n_neighbors in enumerate(n_neighbors_grid):
                    for j, min_dist in enumerate(min_dist_grid):
                        args = dict(
                            n_components=2,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric=metric,
                        )
                        result = fit_results[str(args)]
                        ax[i, j].scatter(
                            *result["reducer"].embedding_.T, s=1, alpha=0.5
                        )
                        ax[i, j].set_title(
                            f'score={result["val_score"]:.2f}'
                        )
                plt.show()

    return fit_results


def umap_train_validate(
    X_train,
    X_val=None,
    model=UMAPEmbedding,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=None,
    verbose=True,
):
    """
    Train a UMAP model and optionally evaluate on validation data.

    Args:
        X_train (numpy.ndarray): Training data.
        X_val (numpy.ndarray, optional): Validation data.
        model (class): UMAP model class.
        n_neighbors (int): Number of neighbors.
        min_dist (float): Minimum distance.
        n_components (int): Embedding dimension.
        metric (str): Distance metric.
        random_state (int, optional): Random seed.
        verbose (bool): Verbosity flag.

    Returns:
        dict: Trained model, validation score, and fit time.
    """
    reducer = model(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )

    start = time.perf_counter()
    reducer.fit(X_train)
    fit_time = time.perf_counter() - start

    val_score = reducer.score(X_val) if X_val is not None else 0.0

    if verbose:
        print("fit_time:", fit_time)
        print("val_score:", val_score)

    return {
        "reducer": reducer,
        "val_score": val_score,
        "fit_time": fit_time,
    }


def umap_fit(X, n_neighbors=15, min_dist=0.1, n_components=2,
             train_size=0.2, random_state=None):
    """
    Fit a UMAP model using a subsampled training set.

    Args:
        X (numpy.ndarray): Input data.
        n_neighbors (int): Number of neighbors.
        min_dist (float): Minimum distance.
        n_components (int): Embedding dimension.
        train_size (float): Fraction used for training.
        random_state (int, optional): Random seed.

    Returns:
        umap.UMAP: Trained reducer.
    """
    train_idx = np.linspace(
        0, X.shape[0] - 1, int(X.shape[0] * train_size), dtype=int
    )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    reducer.fit(X[train_idx])
    return reducer


def umap_transform(X, reducer):
    """
    Transform data using a fitted UMAP reducer.

    Args:
        X (numpy.ndarray): Input data.
        reducer (umap.UMAP): Trained model.

    Returns:
        numpy.ndarray: Embedded data.
    """
    return reducer.transform(X)


def umap_embedding(
    X_proc,
    min_dist,
    n_neighbors,
    n_components=2,
    train_size=0.1,
    random_state=None,
    autoencoder=False,
):
    """
    Compute UMAP embeddings for multiple trials.

    Args:
        X_proc (list): List of feature arrays.
        min_dist (float): Minimum distance.
        n_neighbors (int): Number of neighbors.
        n_components (int): Embedding dimension.
        train_size (float): Training fraction.
        random_state (int, optional): Random seed.
        autoencoder (bool): Use AutoEncoderUMAP.

    Returns:
        tuple:
            - embedded_trials (list)
            - reducer (UMAP model)
    """
    X_proc_concat = np.concatenate(X_proc, axis=0, dtype="float32")
    print("Fitting umap embedding ...")

    if autoencoder:
        X_train_val = X_proc_concat[::int(1 / train_size)]
        X_train, X_val = train_test_split(X_train_val, test_size=0.2)

        reducer = AutoEncoderUMAP(
            input_dim=(X_train.shape[1],),
            min_dist=min_dist,
            n_neighbors=n_neighbors,
            random_state=random_state,
            reconstruction_validation=X_val,
            n_training_epochs=2,
            n_components=n_components,
        )
        reducer.fit(X_train)
    else:
        reducer = UMAPEmbedding(
            min_dist=min_dist,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_components=n_components,
        )
        reducer.fit(X_proc_concat, train_size=train_size)

    print("Embedding into UMAP ...")
    X_embedded = [reducer.transform(X) for X in X_proc]

    return X_embedded, reducer


class SpatialClustering:
    """
    Perform spatial clustering on embedded features using KDE and watershed.
    """

    def __init__(self, bw, nb_gridpoints, watershed_threshold):
        """
        Initialize spatial clustering parameters.

        Args:
            bw (float): KDE bandwidth.
            nb_gridpoints (int): Grid resolution.
            watershed_threshold (float): Density threshold.
        """
        self.bw = bw
        self.nb_gridpoints = nb_gridpoints
        self.watershed_threshold = watershed_threshold

    def fit(self, X_embedded):
        """
        Fit clustering on embedded features.

        Args:
            X_embedded (list): Embedded trials.

        Returns:
            tuple: KDE, grid positions, labels, edges, edge positions.
        """
        X_embedded_concat = np.concatenate(X_embedded, axis=0)
        X_embedded_concat[np.isnan(X_embedded_concat)] = 0

        positions, Z = kernel_density_estimation(
            X_embedded_concat, self.bw, self.nb_gridpoints
        )

        labels, labels_edge = watershed_segmentation(
            Z, threshold_factor=self.watershed_threshold
        )

        binsize_x = (positions[:, 0].max() - positions[:, 0].min()) / labels.shape[1]
        binsize_y = (positions[:, 1].max() - positions[:, 1].min()) / labels.shape[0]

        labels_edge_indices = np.array(np.where(labels_edge)).T
        labels_edge_positions = (
            labels_edge_indices[:, [1, 0]] * np.array([[binsize_x, binsize_y]])
            + positions.min(0)
        )

        print(f"[INFO] {len(np.unique(labels)) - 1} unique segments found")

        return Z, positions, labels, labels_edge, labels_edge_positions

    def transform(self, X_embedded, positions, labels):
        """
        Assign embedded points to spatial clusters.

        Args:
            X_embedded (list): Embedded trials.
            positions (numpy.ndarray): KDE grid.
            labels (numpy.ndarray): Cluster labels.

        Returns:
            tuple: KDEs, positions, assigned segment labels.
        """
        X_kde, X_kde_pos, X_segments = [], [], []

        for trial in X_embedded:
            kde_pos, Z = kernel_density_estimation(trial, self.bw, positions)
            X_kde.append(Z)
            X_kde_pos.append(kde_pos)
            X_segments.append(
                assign_segment_to_features_new(trial, positions, labels)
            )

        return X_kde, X_kde_pos, X_segments

def kernel_density_estimation(embedded_poses, bw, gridpoints=None):
    """
    Perform 2D kernel density estimation on embedded data.

    Args:
        embedded_poses (numpy.ndarray): Embedded feature coordinates.
        bw (float): KDE bandwidth.
        gridpoints (int or numpy.ndarray, optional): Grid resolution or grid.

    Returns:
        tuple:
            - positions (numpy.ndarray): Grid positions.
            - Z (numpy.ndarray): Density estimate.
    """
    if gridpoints is None or isinstance(gridpoints, int):
        positions, Z = FFTKDE(kernel="gaussian", bw=bw).fit(
            embedded_poses
        ).evaluate(gridpoints)
    else:
        positions = gridpoints
        Z = FFTKDE(kernel="gaussian", bw=bw).fit(
            embedded_poses
        ).evaluate(gridpoints)

    nb_gridpoints = len(np.unique(positions[:, 0]))
    Z = Z.reshape(nb_gridpoints, nb_gridpoints, order="F")
    return positions, Z


def watershed_segmentation(Z, threshold_factor=0.01):
    """
    Segment a density map using watershed segmentation.

    Args:
        Z (numpy.ndarray): Density map.
        threshold_factor (float): Density threshold.

    Returns:
        tuple:
            - labels (numpy.ndarray): Segment labels.
            - labels_edge (numpy.ndarray): Segment boundaries.
    """
    threshold = Z > threshold_factor
    localMaxIdx = peak_local_max(Z, min_distance=2, labels=threshold)

    localMax = np.zeros_like(Z, dtype=bool)
    for i, j in localMaxIdx:
        localMax[i, j] = True

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-Z, markers, mask=threshold)

    labels_edge = np.array(
        np.sqrt(np.gradient(labels)[0] ** 2 + np.gradient(labels)[1] ** 2),
        dtype=bool,
    )

    return labels, labels_edge


def assign_segment_to_pose(embedded_poses, labels, positions):
    """
    Assign pose trajectories to spatial segments.

    Args:
        embedded_poses (numpy.ndarray): Embedded poses.
        labels (numpy.ndarray): Segment labels.
        positions (numpy.ndarray): Grid positions.

    Returns:
        numpy.ndarray: Segment labels per pose.
    """
    embedded_poses_segment = np.zeros(embedded_poses.shape[:-1], dtype=int)
    labels_flattened = labels.flatten("F")

    positions_array = positions.reshape(
        labels.shape[0], labels.shape[1], positions.shape[-1]
    )

    binsize_x = np.mean(np.diff(positions_array[:, 0, 0]))
    binsize_y = np.mean(np.diff(positions_array[0, :, 1]))

    for position, label in zip(positions, labels_flattened):
        if label != 0:
            for fly_idx in range(embedded_poses.shape[1]):
                embedded_poses_segment[
                    (
                        (position[0] - binsize_x / 2 <= embedded_poses[:, fly_idx, 0])
                        & (position[0] + binsize_x / 2 > embedded_poses[:, fly_idx, 0])
                        & (position[1] - binsize_y / 2 <= embedded_poses[:, fly_idx, 1])
                        & (position[1] + binsize_y / 2 > embedded_poses[:, fly_idx, 1])
                    ),
                    fly_idx,
                ] = label

    return embedded_poses_segment


def assign_segment_to_features(embedded_features, positions, labels):
    """
    Assign embedded feature points to spatial segments.

    Args:
        embedded_features (numpy.ndarray): Embedded features.
        positions (numpy.ndarray): KDE grid.
        labels (numpy.ndarray): Segment labels.

    Returns:
        numpy.ndarray: Segment label per feature.
    """
    embedded_features_segment = np.zeros(embedded_features.shape[0], dtype=int)
    labels_flattened = labels.flatten("F")

    binsize_x = (positions[:, 0].max() - positions[:, 0].min()) / labels.shape[1]
    binsize_y = (positions[:, 1].max() - positions[:, 1].min()) / labels.shape[0]

    for position, label in zip(positions, labels_flattened):
        if label != 0:
            embedded_features_segment[
                (
                    (position[0] - binsize_x / 2 <= embedded_features[:, 0])
                    & (position[0] + binsize_x / 2 > embedded_features[:, 0])
                    & (position[1] - binsize_y / 2 <= embedded_features[:, 1])
                    & (position[1] + binsize_y / 2 > embedded_features[:, 1])
                )
            ] = label

    return embedded_features_segment


def assign_segment_to_features_new(embedded_features, positions, labels):
    """
    Efficiently assign features to segments using bin indexing.

    Args:
        embedded_features (numpy.ndarray): Embedded features.
        positions (numpy.ndarray): KDE grid positions.
        labels (numpy.ndarray): Segment labels.

    Returns:
        numpy.ndarray: Segment label per feature.
    """
    embedded_features_segment = np.zeros(embedded_features.shape[0], dtype=int)
    labels_flattened = labels.flatten("F")

    binsize_x = (positions[:, 0].max() - positions[:, 0].min()) / labels.shape[1]
    binsize_y = (positions[:, 1].max() - positions[:, 1].min()) / labels.shape[0]

    embedded_features_bin = np.zeros((embedded_features.shape[0], 2), dtype="uint")
    embedded_features_bin[:, 0] = np.floor(
        (embedded_features[:, 0] - positions[:, 0].min()) / binsize_x
    )
    embedded_features_bin[:, 1] = np.floor(
        (embedded_features[:, 1] - positions[:, 1].min()) / binsize_y
    )

    for position, label in zip(
        positions[labels_flattened != 0], labels_flattened[labels_flattened != 0]
    ):
        position_bin_x = int((position[0] - positions[:, 0].min()) / binsize_x)
        position_bin_y = int((position[1] - positions[:, 1].min()) / binsize_y)

        embedded_features_segment[
            (embedded_features_bin[:, 0] == position_bin_x)
            & (embedded_features_bin[:, 1] == position_bin_y)
        ] = label

    return embedded_features_segment


def get_umap_segment_labels(X, reducer, positions, labels):
    """
    Assign UMAP segment labels to input features.

    Args:
        X (numpy.ndarray): Input features.
        reducer (UMAPEmbedding): Trained UMAP model.
        positions (numpy.ndarray): KDE grid.
        labels (numpy.ndarray): Segment labels.

    Returns:
        numpy.ndarray: Segment labels.
    """
    X_embedded = reducer.transform(X)
    return assign_segment_to_features_new(X_embedded, positions, labels)


def get_segment_poselets(poselets, embedded_poses_segment):
    """
    Aggregate poselets by segment.

    Args:
        poselets (numpy.ndarray): Poselets.
        embedded_poses_segment (numpy.ndarray): Segment labels.

    Returns:
        tuple: Poselets, means, and standard deviations per segment.
    """
    segment_poselets = {}
    average_segment_poselet = {}
    std_segment_poselet = {}

    for seg_idx in np.unique(embedded_poses_segment):
        segment_poselets[seg_idx] = poselets[embedded_poses_segment == seg_idx]
        average_segment_poselet[seg_idx] = np.nanmean(
            segment_poselets[seg_idx], axis=0
        )
        std_segment_poselet[seg_idx] = np.nanstd(
            segment_poselets[seg_idx], axis=0
        )

    return segment_poselets, average_segment_poselet, std_segment_poselet


def get_segment_pose_positions(pose_positions, embedded_poses_segment=None):
    """
    Aggregate pose positions by segment.

    Args:
        pose_positions (numpy.ndarray): Pose positions.
        embedded_poses_segment (numpy.ndarray): Segment labels.

    Returns:
        tuple: Segment poses, mean poses, and standard deviations.
    """
    segment_pose_positions = {}
    segment_pose_positions_avg = {}
    segment_pose_positions_std = {}

    for seg_idx in np.unique(embedded_poses_segment):
        segment_pose_positions[seg_idx] = pose_positions[
            embedded_poses_segment == seg_idx
        ]
        segment_pose_positions_avg[seg_idx] = np.nanmean(
            segment_pose_positions[seg_idx], axis=0
        )
        segment_pose_positions_std[seg_idx] = np.nanstd(
            segment_pose_positions[seg_idx], axis=0
        )

    return (
        segment_pose_positions,
        segment_pose_positions_avg,
        segment_pose_positions_std,
    )


def plot_fly_skeleton(pose_position, poseparts, axis):
    """
    Plot a fly skeleton for a single frame.

    Args:
        pose_position (numpy.ndarray): Pose coordinates.
        poseparts (array-like): Pose part names.
        axis (matplotlib.axes.Axes): Axis to plot on.

    Returns:
        tuple: Scatter and line objects.
    """
    axis.cmap = plt.cm.rainbow
    pose_scat = axis.scatter(pose_position[0], pose_position[1], 10)
    pose_connector = []

    for part in range(len(poseparts)):
        pose_connector.append(
            axis.plot(
                pose_position[0][[part, 8]],
                pose_position[1][[part, 8]],
                c="k",
                lw=0.5,
            )
        )

    return pose_scat, pose_connector


def animate_fly_skeleton(pose_position, poseparts, pose_scat, pose_connector):
    """
    Update fly skeleton plot for animation.

    Args:
        pose_position (numpy.ndarray): Current pose.
        poseparts (array-like): Pose part names.
        pose_scat: Scatter object.
        pose_connector: Line objects.

    Returns:
        tuple: Updated artists.
    """
    pose_scat.set_offsets(pose_position)

    for part in range(len(poseparts)):
        pose_connector[part].set_data(
            pose_position[[part, 8], 0], pose_position[[part, 8], 1]
        )

    return pose_scat, pose_connector


def slope(x1, y1, x2, y2):
    """
    Compute slope of a line.

    Args:
        x1, y1, x2, y2 (float): Coordinates.

    Returns:
        float: Slope.
    """
    return (y2 - y1) / (x2 - x1)


def angle_between_lines(lineA, lineB):
    """
    Compute angle between two lines.

    Args:
        lineA (tuple): Two points defining line A.
        lineB (tuple): Two points defining line B.

    Returns:
        float: Angle in degrees.
    """
    m1 = slope(*lineA[0], *lineA[1])
    m2 = slope(*lineB[0], *lineB[1])
    return np.rad2deg(np.arctan((m2 - m1) / (1 + m2 * m1)))


def compute_gradients(embedding, nb_gridpoints=None,
                      xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Compute gradient magnitudes and directions in embedding space.

    Args:
        embedding (numpy.ndarray): Embedded trajectory.
        nb_gridpoints (int, optional): Grid resolution.
        xmin, xmax, ymin, ymax (float, optional): Spatial bounds.

    Returns:
        tuple: Gradient magnitudes, directions, and optional grid bins.
    """
    gradients = np.diff(embedding, axis=0)

    if nb_gridpoints is None:
        return (
            np.linalg.norm(gradients, axis=1),
            np.rad2deg(np.arctan(gradients)),
        )

    xbins = np.linspace(
        xmin if xmin is not None else embedding[:, 0].min(),
        xmax if xmax is not None else embedding[:, 0].max(),
        nb_gridpoints + 1,
    )
    ybins = np.linspace(
        ymin if ymin is not None else embedding[:, 1].min(),
        ymax if ymax is not None else embedding[:, 1].max(),
        nb_gridpoints + 1,
    )

    gradient_directions = np.full((nb_gridpoints, nb_gridpoints), np.nan)
    gradient_magnitudes = np.full((nb_gridpoints, nb_gridpoints), np.nan)

    for i in range(nb_gridpoints):
        for j in range(nb_gridpoints):
            idx = np.where(
                (embedding[1:, 0] >= xbins[i])
                & (embedding[1:, 0] < xbins[i + 1])
                & (embedding[1:, 1] >= ybins[j])
                & (embedding[1:, 1] < ybins[j + 1])
            )[0]
            if len(idx) > 0:
                g = gradients[idx]
                gradient_magnitudes[i, j] = np.nanmedian(np.linalg.norm(g, axis=1))
                gradient_directions[i, j] = np.nanmedian(
                    np.rad2deg(np.arctan(g))
                )

    return gradient_magnitudes, gradient_directions, xbins, ybins


def rotate(origin, point, angle):
    """
    Rotate a point around an origin.

    Args:
        origin (tuple): Origin point.
        point (tuple): Point to rotate.
        angle (float): Rotation angle in radians.

    Returns:
        tuple: Rotated point.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def plot_gradients(gradient_directions, gradient_magnitudes,
                   xbins=None, ybins=None, ax=None):
    """
    Plot gradient vectors.

    Args:
        gradient_directions (numpy.ndarray): Gradient directions.
        gradient_magnitudes (numpy.ndarray): Gradient magnitudes.
        xbins (numpy.ndarray, optional): X bins.
        ybins (numpy.ndarray, optional): Y bins.
        ax (matplotlib.axes.Axes, optional): Axis to plot on.
    """
    if ax is None:
        _, ax = plt.subplots()

    if xbins is None:
        xbins = np.arange(gradient_directions.shape[0] + 1)
    if ybins is None:
        ybins = np.arange(gradient_directions.shape[1] + 1)

    for i in range(len(xbins) - 1):
        for j in range(len(ybins) - 1):
            angle = gradient_directions[i, j]
            if np.isnan(angle):
                continue

            cx = 0.5 * (xbins[i] + xbins[i + 1])
            cy = 0.5 * (ybins[j] + ybins[j + 1])
            dx, dy = rotate(
                (cx, cy),
                (cx + gradient_magnitudes[i, j], cy),
                np.deg2rad(angle),
            )
            ax.arrow(cx, cy, dx - cx, dy - cy, width=0.1)


def edit_colormap(base_cmap, ncolors=128,
                  edit_indices=0, edit_colors=[1, 1, 1, 1]):
    """
    Modify a matplotlib colormap.

    Args:
        base_cmap (str): Base colormap name.
        ncolors (int): Number of colors.
        edit_indices (int or list): Indices to modify.
        edit_colors (list): RGBA replacement color.

    Returns:
        matplotlib.colors.ListedColormap: Modified colormap.
    """
    newcolors = mpl.colormaps[base_cmap](np.linspace(0, 1, ncolors))
    newcolors[edit_indices] = edit_colors
    return mpl.colors.ListedColormap(newcolors)
