# -*- coding: utf-8 -*-
"""
FTCP-VAE Training and Evaluation Pipeline

Author: Issa Onishi
Created: October 1, 2026
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
# Enable eager execution for TF 1.15
tf.compat.v1.enable_eager_execution()
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from module.data import FTCP_represent
from module.utils import pad, minmax_fit, minmax_transform, inv_minmax
from module.FTCP import LossHistoryCallback, FTCP_VAE


# =============================================================================
# Experiment settings
# =============================================================================
data_name = 'data_query_3_5_elements_property_nsites_below_112'

# Structural parameters for 3–5-element compounds
max_elms, min_elms, max_sites = 5, 3, 112

# Specify the CNN model to be trained and the network architecture
network_pattern_numbers = ["0", "1", "2", "3", "4", "5", "6"]
cnn_patterns            = ['0', '1', '2']
supervised_modes        = [True, False]

# SVAE/USVAE taining hyperparameters
epochs        = 200
batch_size    = 256
learning_rate = 5e-4
coeff_KL      = 2
coeff_prop    = 10

# Execution flags
restart_flag     = True    # Whether to restart training from an intermediate state
code_test        = False   # For testing code. Set to False for production runs.

# Target properties
prop = ['formation_energy_per_atom', 'band_gap']


# =============================================================================
# Data loading and preprocessing
# =============================================================================
dataframe = pd.read_csv(f'./{data_name}.csv', index_col=0)
if code_test:
    dataframe = dataframe.sample(min(100, len(dataframe)), random_state=42)

# Generate FTCP crystal structure tensors (raw, not yet normalized).
#
# This is cached to FTCP_representation_full.npz: if a cache from a previous
# run already exists and matches the current `dataframe` (same row count),
# it is loaded instead of recomputing FTCP_represent() (which parses every
# CIF and is expensive), so re-running main.py doesn't redo this step every
# time. The same cache is also reused by learning_result_analysis.py.
os.makedirs(f'./{data_name}_result', exist_ok=True)
ftcp_cache_path = f'./{data_name}_result/FTCP_representation_full.npz'

FTCP_representation = None
if os.path.exists(ftcp_cache_path):
    print(f'Found cached FTCP representation: {ftcp_cache_path}')
    _cache = np.load(ftcp_cache_path)
    _cached_FTCP, _cached_Nsites = _cache['FTCP_representation'], _cache['Nsites']
    if len(_cached_FTCP) == len(dataframe):
        print('  Row count matches current dataframe; using cache (skipping FTCP_represent).')
        FTCP_representation, Nsites = _cached_FTCP, _cached_Nsites
    else:
        print(
            f'  WARNING: cache has {len(_cached_FTCP)} rows but dataframe has '
            f'{len(dataframe)} rows (code_test flag or input CSV may have changed). '
            'Recomputing.'
        )

if FTCP_representation is None:
    FTCP_representation, Nsites = FTCP_represent(dataframe, max_elms, max_sites, return_Nsites=True)
    FTCP_representation = pad(FTCP_representation, 2)
    np.savez_compressed(
        ftcp_cache_path,
        FTCP_representation=FTCP_representation,
        Nsites=Nsites,
    )

Y_raw = dataframe[prop].values.astype('float32')

# -----------------------------------------------------------------------
# Train / validation / test split
# -----------------------------------------------------------------------
# Split off the test set first (identical composition to the previous
# 80/20 split: same test_size and random_state), then carve a validation
# set out of the remaining 80%. 0.125 * 0.8 = 0.1, so the resulting split
# is 70% train / 10% validation / 20% test.
#
# The validation set is used to compare the 42 architecture combinations
# and select the best one (Table 3, S1, S2). The test set is held out and
# used only once, for the final evaluation of the selected model
# (Table 4-6), so that model selection never touches the test set.
ind_trainval, ind_test = train_test_split(np.arange(len(Y_raw)), test_size=0.2, random_state=21)
ind_train, ind_val = train_test_split(ind_trainval, test_size=0.125, random_state=21)

# -----------------------------------------------------------------------
# Fit normalization on the TRAINING subset only, then apply (transform
# only, no refitting) to train/val/test. This avoids validation/test
# information leaking into the min/max normalization parameters.
# -----------------------------------------------------------------------
scaler_X = minmax_fit(FTCP_representation[ind_train].astype('float32'))
X_train  = minmax_transform(FTCP_representation[ind_train].astype('float32'), scaler_X)
X_val    = minmax_transform(FTCP_representation[ind_val].astype('float32'),   scaler_X)
X_test   = minmax_transform(FTCP_representation[ind_test].astype('float32'),  scaler_X)

scaler_y = MinMaxScaler()
scaler_y.fit(Y_raw[ind_train])
y_train = scaler_y.transform(Y_raw[ind_train]).astype('float32')
y_val   = scaler_y.transform(Y_raw[ind_val]).astype('float32')
y_test  = scaler_y.transform(Y_raw[ind_test]).astype('float32')

# Persist the split indices and fitted scalers so that downstream analysis
# scripts (e.g. learning_result_analysis.py) reuse the exact same
# train-only-fitted normalization instead of re-deriving their own.
os.makedirs(f'./{data_name}_result', exist_ok=True)
joblib.dump(
    {'ind_train': ind_train, 'ind_val': ind_val, 'ind_test': ind_test},
    f'./{data_name}_result/split_indices.pkl',
)
joblib.dump(scaler_X, f'./{data_name}_result/scaler_X.pkl')
joblib.dump(scaler_y, f'./{data_name}_result/scaler_y.pkl')

# Load element list
elm_str = joblib.load('data/element.pkl')


# =============================================================================
# Evaluation functions
# =============================================================================
def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true + 1e-12), np.array(y_pred + 1e-12)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MAE(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred), axis=0)

def MAE_site_coor(SITE_COOR, SITE_COOR_recon, Nsites):
    """MAE for site coordinates, evaluated only over occupied sites."""
    site, site_recon = [], []
    for i in range(len(SITE_COOR)):
        site.append(SITE_COOR[i, :Nsites[i], :])
        site_recon.append(SITE_COOR_recon[i, :Nsites[i], :])
    site      = np.vstack(site)
    site_recon = np.vstack(site_recon)
    return np.mean(np.ravel(np.abs(site - site_recon)))


# =============================================================================
# Training and evaluation loop
# =============================================================================
for cnn_pattern, supervised in product(cnn_patterns, supervised_modes):
    # Set output directory based on VAE type
    vae_type = "SVAE" if supervised else "USVAE"
    base_dir = f'./{data_name}_result/CNN_{cnn_pattern}/{vae_type}_result'

    subdirs = {
        "result_dir":       os.path.join(base_dir, "recon_result"),
        "latent_dir":       os.path.join(base_dir, "latent_variable"),
        "learning_log_dir": os.path.join(base_dir, "learning_log"),
    }
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    learning_log_dir = subdirs["learning_log_dir"]

    for network_pattern_number in network_pattern_numbers:

        # Output file paths
        csv_path          = f"{learning_log_dir}/loss_pattern{network_pattern_number}.csv"
        weight_model_name = f"{learning_log_dir}/VAE_weight_pattern{network_pattern_number}"

        # -----------------------------------------------------------------
        # Model definition and compilation
        # -----------------------------------------------------------------
        vae_model = FTCP_VAE(
            X_train=X_train,
            y_train=y_train,
            supervised=supervised,
            coeff_KL=coeff_KL,
            coeff_prop=coeff_prop,
            restart=restart_flag,
            network_pattern=network_pattern_number,
            cnn_pattern=cnn_pattern,
            csv_path=csv_path,
            model_prefix=weight_model_name,
        )

        vae_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=lambda y_true, y_pred: 0.0
        )

        # -----------------------------------------------------------------
        # Callbacks
        # -----------------------------------------------------------------
        loss_history = LossHistoryCallback(
            csv_path=csv_path,
            loss_plot_dir=None,      # loss curves are not used in the paper
            save_plot_loss=False,    # -> no PNG is generated (CSV only)
            restart=(restart_flag and vae_model.start_epoch > 0),
            supervised=supervised
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='loss', factor=0.3, patience=4, min_lr=1e-6
        )

        def scheduler(epoch, lr):
            if epoch == 50:   return 1e-4
            if epoch == 100:  return 5e-5
            return lr

        schedule_lr = LearningRateScheduler(scheduler)

        # -----------------------------------------------------------------
        # Training
        # -----------------------------------------------------------------
        if supervised:
            vae_model.fit(
                x=(X_train, y_train),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[loss_history, reduce_lr, schedule_lr]
            )
        else:
            vae_model.fit(
                x=X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[loss_history, reduce_lr, schedule_lr]
            )

        # -----------------------------------------------------------------
        # Reconstruction and evaluation
        # -----------------------------------------------------------------
        def evaluate_reconstruction(X_eval, y_eval, Nsites_eval):
            """Evaluate reconstruction/property-prediction metrics on a given
            (already-normalized) subset, using the scaler fitted on the
            training data only (scaler_X / scaler_y)."""
            if supervised:
                X_eval_recon = vae_model.predict([X_eval, y_eval], verbose=0)
            else:
                X_eval_recon = vae_model.predict(X_eval, verbose=0)

            X_eval_       = inv_minmax(X_eval, scaler_X)
            X_eval_recon_ = inv_minmax(X_eval_recon, scaler_X)

            n_elm = len(elm_str)

            abc       = X_eval_[:, n_elm, :3]
            abc_recon = X_eval_recon_[:, n_elm, :3]
            mape_abc  = MAPE(abc, abc_recon)

            ang       = X_eval_[:, n_elm + 1, :3]
            ang_recon = X_eval_recon_[:, n_elm + 1, :3]
            mape_ang  = MAPE(ang, ang_recon)

            coor       = X_eval_[:, n_elm + 2:n_elm + 2 + max_sites, :3]
            coor_recon = X_eval_recon_[:, n_elm + 2:n_elm + 2 + max_sites, :3]
            mae_coor   = MAE_site_coor(coor, coor_recon, Nsites_eval)

            elm_accu = []
            for i in range(max_elms):
                elm       = np.argmax(X_eval_[:, :n_elm, i], axis=1)
                elm_recon = np.argmax(X_eval_recon_[:, :n_elm, i], axis=1)
                elm_accu.append(metrics.accuracy_score(elm, elm_recon))
            mean_accu = np.mean(elm_accu)

            if supervised:
                y_eval_hat  = vae_model.predict_y(X_eval, verbose=0)
                y_eval_     = scaler_y.inverse_transform(y_eval)
                y_eval_hat_ = scaler_y.inverse_transform(y_eval_hat)
                mae_ef, mae_eg = MAE(y_eval_, y_eval_hat_)
            else:
                mae_ef, mae_eg = np.nan, np.nan

            return pd.DataFrame({
                'MAE Ef (eV)':               [mae_ef],
                'MAE Eg (eV)':               [mae_eg],
                'Element accuracy':           [mean_accu],
                'Lattice constant MAPE (%)':  [mape_abc],
                'Lattice angle MAPE (%)':     [mape_ang],
                'Site coordinate MAE (frac)': [mae_coor],
            })

        # Validation-set evaluation: this is the metric used to COMPARE the
        # 42 architecture combinations and select the best one (Table 3,
        # S1, S2). The test set is intentionally not used here, so that
        # model selection never sees the final test set.
        val_result_df = evaluate_reconstruction(X_val, y_val, Nsites[ind_val])
        print(f"[Validation] Lattice constant MAPE: {val_result_df['Lattice constant MAPE (%)'].iloc[0]:.4f} %")
        print(f"[Validation] Lattice angle MAPE: {val_result_df['Lattice angle MAPE (%)'].iloc[0]:.4f} %")
        print(f"[Validation] Site coordinate MAE: {val_result_df['Site coordinate MAE (frac)'].iloc[0]:.6f} (fractional)")
        val_result_df.to_csv(
            os.path.join(subdirs["result_dir"], f"FTCP_evaluation_pattern{network_pattern_number}.csv"),
            index=True
        )

        # Test-set evaluation: computed for every combination for
        # transparency/logging, but this file must NEVER be used to choose
        # among architectures. Only the finally selected configuration's
        # test-set numbers (from this file) are reported in the paper as
        # the final evaluation (Table 4-6).
        test_result_df = evaluate_reconstruction(X_test, y_test, Nsites[ind_test])
        print(f"[Test] Lattice constant MAPE: {test_result_df['Lattice constant MAPE (%)'].iloc[0]:.4f} %")
        print(f"[Test] Lattice angle MAPE: {test_result_df['Lattice angle MAPE (%)'].iloc[0]:.4f} %")
        print(f"[Test] Site coordinate MAE: {test_result_df['Site coordinate MAE (frac)'].iloc[0]:.6f} (fractional)")
        test_result_df.to_csv(
            os.path.join(subdirs["result_dir"], f"FTCP_evaluation_TEST_pattern{network_pattern_number}.csv"),
            index=True
        )