# -*- coding: utf-8 -*-
"""
FTCP-VAE Training and Evaluation Pipeline

Author: Issa Onishi
Created: September 18, 2025
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
from module.data import data_query, FTCP_represent
from module.utils import pad, minmax, inv_minmax
from module.FTCP import LossHistoryCallback, FTCP_VAE


# =============================================================================
# Experiment settings
# =============================================================================
data_name = 'data_query_3_5_elements_property_nsites_below_112'

# Structural parameters for 3–5-element compounds
max_elms, min_elms, max_sites = 5, 3, 112

# Specify the CNN model to be trained and the network architecture
network_pattern_numbers = ["0", "1", "2", "3", "4", "5", "6"]
cnn_patterns            = ['0', '1', '2', '3']
supervised_modes        = [True, False]

# SVAE/USVAE taining hyperparameters
epochs        = 200
batch_size    = 256
learning_rate = 5e-4
coeff_KL      = 2
coeff_prop    = 10

# Execution flags
restart_flag     = True    # Whether to restart training from an intermediate state
code_test        = True   # For testing code

# Target properties
prop = ['formation_energy_per_atom', 'band_gap']


# =============================================================================
# Data loading and preprocessing
# =============================================================================
dataframe = pd.read_csv(f'./{data_name}.csv', index_col=0)
if code_test:
    dataframe = dataframe.sample(min(100, len(dataframe)), random_state=42)

# Generate FTCP crystal structure tensors
FTCP_representation, Nsites = FTCP_represent(dataframe, max_elms, max_sites, return_Nsites=True)
FTCP_representation = pad(FTCP_representation, 2)
X, scaler_X = minmax(FTCP_representation.astype('float32'))

# Extract and normalize property labels
scaler_y = MinMaxScaler()
Y = scaler_y.fit_transform(dataframe[prop].values).astype('float32')

# Train/test split
ind_train, ind_test = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=21)
X_train, X_test = X[ind_train], X[ind_test]
y_train, y_test = Y[ind_train], Y[ind_test]

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
        loss_plot_dir     = f"{learning_log_dir}/plot_pattern{network_pattern_number}"
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
            loss_plot_dir=loss_plot_dir,
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
        # Latent space extraction and visualization
        # -----------------------------------------------------------------
        train_latent = vae_model.compress_to_latent(X_train, verbose=1)
        y_train_     = scaler_y.inverse_transform(y_train)
        y_test_      = scaler_y.inverse_transform(y_test)

        font_size = 26
        plt.rcParams['axes.labelsize']  = font_size
        plt.rcParams['xtick.labelsize'] = font_size - 2
        plt.rcParams['ytick.labelsize'] = font_size - 2
        
        print('=============== Generating latent space plots ===============')
        fig, ax = plt.subplots(1, 2, figsize=(18, 7.3))
        fig.text(0.016, 0.92, '(A) $E_\\mathrm{f}$', fontsize=font_size)
        fig.text(0.533, 0.92, '(B) $E_\\mathrm{g}$', fontsize=font_size)

        s0 = ax[0].scatter(train_latent[:, 0], train_latent[:, 2],
                            s=7, c=np.squeeze(y_train_[:, 0]), cmap='viridis')
        plt.colorbar(s0, ax=ax[0]).set_label('$E_f$ (eV)')

        s1 = ax[1].scatter(train_latent[:, 0], train_latent[:, 2],
                            s=7, c=np.squeeze(y_train_[:, 1]), cmap='viridis')
        plt.colorbar(s1, ax=ax[1]).set_label('$E_g$ (eV)')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, top=0.85)
        plt.savefig(
            os.path.join(subdirs["result_dir"], f"Ef_Eg_pattern{network_pattern_number}.png"),
            bbox_inches="tight", dpi=300
        )
        plt.close()

        # -----------------------------------------------------------------
        # Reconstruction and evaluation
        # -----------------------------------------------------------------
        if supervised:
            X_test_recon = vae_model.predict([X_test, y_test], verbose=0)
        else:
            X_test_recon = vae_model.predict(X_test, verbose=0)

        X_test_       = inv_minmax(X_test, scaler_X)
        X_test_recon_ = inv_minmax(X_test_recon, scaler_X)
        X_test_recon_[X_test_recon_ < 0.1] = 0

        n_elm = len(elm_str)

        # Lattice constants and angles
        abc       = X_test_[:, n_elm, :3]
        abc_recon = X_test_recon_[:, n_elm, :3]
        mape_abc  = MAPE(abc, abc_recon)
        print(f"Lattice constant MAPE: {mape_abc:.4f} %")

        ang       = X_test_[:, n_elm + 1, :3]
        ang_recon = X_test_recon_[:, n_elm + 1, :3]
        mape_ang  = MAPE(ang, ang_recon)
        print(f"Lattice angle MAPE: {mape_ang:.4f} %")

        # Site coordinates
        coor       = X_test_[:, n_elm + 2:n_elm + 2 + max_sites, :3]
        coor_recon = X_test_recon_[:, n_elm + 2:n_elm + 2 + max_sites, :3]
        mae_coor   = MAE_site_coor(coor, coor_recon, Nsites[ind_test])
        print(f"Site coordinate MAE: {mae_coor:.6f} (fractional)")

        # Element classification accuracy
        elm_accu = []
        for i in range(max_elms):
            elm       = np.argmax(X_test_[:, :n_elm, i], axis=1)
            elm_recon = np.argmax(X_test_recon_[:, :n_elm, i], axis=1)
            elm_accu.append(metrics.accuracy_score(elm, elm_recon))
        mean_accu = np.mean(elm_accu)
        print(f"Element classification accuracy: {elm_accu}")

        # Property prediction (supervised only)
        if supervised:
            y_test_hat  = vae_model.predict_y(X_test, verbose=1)
            y_test_hat_ = scaler_y.inverse_transform(y_test_hat)
            mae_ef, mae_eg = MAE(y_test_, y_test_hat_)
            print(f"Property MAE: Ef={mae_ef:.6f} eV, Eg={mae_eg:.6f} eV")
        else:
            mae_ef, mae_eg = np.nan, np.nan
            print("Unsupervised mode: property prediction disabled")

        # Save evaluation results
        result_df = pd.DataFrame({
            'MAE Ef (eV)':               [mae_ef],
            'MAE Eg (eV)':               [mae_eg],
            'Element accuracy':           [mean_accu],
            'Lattice constant MAPE (%)':  [mape_abc],
            'Lattice angle MAPE (%)':     [mape_ang],
            'Site coordinate MAE (frac)': [mae_coor],
        })
        result_df.to_csv(
            os.path.join(subdirs["result_dir"], f"FTCP_evaluation_pattern{network_pattern_number}.csv"),
            index=True
        )