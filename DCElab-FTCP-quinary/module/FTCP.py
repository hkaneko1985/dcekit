# -*- coding: utf-8 -*-
"""
FTCP-VAE Training and Evaluation Pipeline

Author: Issa Onishi
Created: October 1, 2026
"""

import os, sys, time, random, gc, tracemalloc
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from module.vae.encoder import EncoderPattern
from module.vae.decoder import DecoderPattern



class LossHistoryCallback(tf.keras.callbacks.Callback):
    """
    Reads loss from logs at the end of each epoch, saves to CSV and visualizes.
    loss_prop is handled only when supervised=True.
    """
    def __init__(self,
                 csv_path=None,
                 loss_plot_dir=None,
                 save_plot_loss=True,
                 restart=True,
                 supervised=True):
        super().__init__()
        self.csv_path = csv_path
        self.loss_plot_dir = loss_plot_dir
        self.save_plot_loss = save_plot_loss
        self.supervised = supervised

        # Create directory for loss plots
        if loss_plot_dir:
            os.makedirs(loss_plot_dir, exist_ok=True)

        # Choose whether to retain existing CSV or create a new one
        self.restart = restart
        if csv_path:
            if restart and os.path.exists(csv_path):
                self.df = pd.read_csv(csv_path)
            else:
                # Switch column layout based on supervised/unsupervised
                base_cols = ["epoch", "loss_total", "loss_recon", "loss_KL"]
                if supervised:
                    base_cols.append("loss_prop")
                self.df = pd.DataFrame(columns=base_cols)
                self.df.to_csv(csv_path, index=False)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Increment epoch from the last recorded value
        next_epoch = int(self.df["epoch"].iloc[-1]) + 1 if len(self.df) > 0 else 1

        # Update log
        if self.csv_path:
            # Switch columns based on supervised/unsupervised
            row_data = {
                "epoch":      next_epoch,
                "loss_total": float(logs.get("loss", 0)),
                "loss_recon": float(logs.get("recon_loss", 0)),
                "loss_KL":    float(logs.get("kl_loss", 0))
            }
            if self.supervised:
                row_data["loss_prop"] = float(logs.get("prop_loss", 0))

            new_row = pd.DataFrame([row_data])
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            self.df.to_csv(self.csv_path, index=False, float_format="%.7f")

        # Plot helper
        def save_plot(column, title):
            plt.rcParams['font.size'] = 28
            plt.rcParams['figure.figsize'] = (11, 8)
            plt.plot(self.df["epoch"], self.df[column], marker="o")
            plt.title(title, pad=20)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            if self.loss_plot_dir:
                plt.savefig(os.path.join(self.loss_plot_dir, f"{column}.png"), dpi=300)
            plt.close()

        # Generate and save plots
        if self.save_plot_loss:
            for col in ["loss_total", "loss_recon", "loss_KL"]:
                if col in self.df.columns:
                    save_plot(col, col)
            if self.supervised and "loss_prop" in self.df.columns:
                save_plot("loss_prop", "loss_prop")


# ==================== FTCP_VAE ====================
class FTCP_VAE(Model):
    """
    FTCP_VAE (TensorFlow implementation)
    - supervised=True  : VAE with property regression head (supervised VAE)
    - supervised=False : Unsupervised VAE
    - Encoder/Decoder architecture is selected by network_pattern
    - Supports weight restart (.keras format)
    - Loss follows the paper implementation (sum then mean ordering)
    """

    def __init__(self, X_train, y_train=None,
                 supervised=True,
                 coeff_KL=2.0, coeff_prop=10.0,
                 random_seed=42,
                 restart=True,
                 csv_path=None, model_prefix=None,
                 network_pattern="0",
                 cnn_pattern="0"):

        super(FTCP_VAE, self).__init__()

        # Reproducibility settings (lightweight; compatible with GPU/CPU)
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            try:
                tf.random.set_seed(random_seed)
            except AttributeError:
                tf.compat.v1.set_random_seed(random_seed)

        # Parameters
        self.supervised       = supervised
        self.network_pattern  = network_pattern
        self.coeff_KL         = coeff_KL
        self.coeff_prop       = coeff_prop if supervised else 0.0

        self.csv_path         = csv_path
        self.model_prefix     = model_prefix
        self.vae_path         = f"{self.model_prefix}.keras" if self.model_prefix else None
        self.restart          = bool(restart and self.csv_path and self.model_prefix)
        self.start_epoch      = 0
        self.history_rows     = []

        # Validate input dimensions
        if X_train.ndim != 3:
            raise ValueError("X_train must be 3D.")
        self.input_dim      = int(X_train.shape[1])
        self.channel_dim    = int(X_train.shape[2])
        self.regression_dim = int(y_train.shape[1]) if (supervised and y_train is not None) else 1

        # ==================== Build Encoder / Decoder ====================
        encoder_builder = EncoderPattern(network_pattern=network_pattern,
                                         cnn_pattern=cnn_pattern)
        self.encoder, self.latent_dim, self.map_size = encoder_builder.build(
            (self.input_dim, self.channel_dim)
        )
        # Store latent_dim as attribute for regression head construction
        latent_dim = self.latent_dim

        decoder_builder = DecoderPattern(network_pattern=network_pattern,
                                         cnn_pattern=cnn_pattern,
                                         map_size=self.map_size,
                                         channel_dim=self.channel_dim)
        self.decoder = decoder_builder.build()

        # ==================== Regression head (supervised only) ====================
        # Fix: regression head takes z_mean directly to avoid double encoder call in train_step
        if self.supervised:
            z_mean_input = tf.keras.Input(shape=(self.latent_dim,), name="z_mean_input")
            r     = tf.keras.layers.Activation('relu')(z_mean_input)
            r     = tf.keras.layers.Dense(128, activation="relu")(r)
            r     = tf.keras.layers.Dense(32,  activation="relu")(r)
            y_hat = tf.keras.layers.Dense(self.regression_dim, activation="sigmoid")(r)
            self.regression = Model(z_mean_input, y_hat, name="regression")
        else:
            self.regression = None

        # ==================== Combine into full VAE model ====================
        # Fix: remove dangling regression_inputs that was never connected to outputs
        z_, z_mean_, z_log_var_ = self.encoder.output
        vae_outputs = self.decoder(z_)
        super(FTCP_VAE, self).__init__(inputs=self.encoder.input, outputs=vae_outputs)

        self.z_mean, self.z_log_var, self.z = z_mean_, z_log_var_, z_

        # ==================== Restart: load saved weights ====================
        if self.restart and self.vae_path and os.path.isfile(self.vae_path):
            try:
                self.load_weights(self.vae_path)
                print(f"[Info] Weights loaded from {self.vae_path}")
                if os.path.exists(self.csv_path):
                    df = pd.read_csv(self.csv_path)
                    if "epoch" in df.columns and len(df) > 0:
                        self.start_epoch  = int(df["epoch"].iloc[-1])
                        self.history_rows = df.to_dict("records")
                        print(f"[Info] Restart enabled. Resume from epoch {self.start_epoch+1}.")
            except Exception as e:
                print("[Error] Could not load restart file:", e)
                if os.path.exists(self.csv_path):
                    os.remove(self.csv_path)
                    print(f"[Info] Corrupted CSV removed: {self.csv_path}")
                self.start_epoch  = 0
                self.history_rows = []


    # ==================== Forward pass ====================
    def call(self, inputs, training=False):
        # Accepts predict([X, y]); y is received but not used.
        x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        z, _, _ = self.encoder(x, training=training)
        x_recon = self.decoder(z, training=training)
        return x_recon


    # ==================== Loss computation ====================
    def compute_loss(self, x, y, x_recon, y_pred, z_mean, z_log_var):
        # Reconstruction loss (sum over all elements, as per original paper)
        loss_recon = K.sum(K.square(x - x_recon))
        # KL divergence (mean over latent dimensions -> shape: (batch,), as per original paper)
        loss_KL_vec = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # Property loss (sum over all elements, as per original paper)
        if self.supervised and (y is not None) and (y_pred is not None):
            loss_prop = K.sum(K.square(y - y_pred))
        else:
            loss_prop = tf.constant(0.0, dtype=tf.float32)
        # Total loss (mean applied last, as per original paper)
        total_loss = K.mean(loss_recon + self.coeff_KL * loss_KL_vec + self.coeff_prop * loss_prop)
        return total_loss, tf.cast(loss_recon, tf.float32), tf.cast(tf.reduce_mean(loss_KL_vec), tf.float32), tf.cast(loss_prop, tf.float32)


    # ==================== Training step ====================
    # Fix: compile train_step into a graph with @tf.function.
    # Currently this method is called as plain eager Python once per batch
    # (module/FTCP.py fit(), "for step, batch in enumerate(dataset): ...
    # self.train_step(batch)"), with no @tf.function anywhere. Keras' own
    # Model.fit() never does this: internally it compiles the per-batch
    # step once (e.g. via _make_train_function()/K.function() in the
    # TF1.15 code actually installed in this project - see
    # tensorflow/tensorflow v1.15.5, tensorflow/python/keras/engine/training.py)
    # and reuses that compiled function for every subsequent batch, instead
    # of re-running the encoder/decoder/regression calls and GradientTape
    # bookkeeping through the Python interpreter every single step.
    # A custom tf.GradientTape training loop that skips this compilation
    # step is a known, hardware-independent way to get per-step/per-epoch
    # time that keeps climbing over a long run - reported independently by
    # multiple users on tensorflow/tensorflow, on TF versions and machines
    # unrelated to each other:
    #   - https://github.com/tensorflow/tensorflow/issues/23014
    #     ("gradient calculation takes gradually more time each epoch",
    #     reproduced on multiple machines and TF versions)
    #   - https://github.com/tensorflow/tensorflow/issues/48732
    #     (custom GradientTape-based fit(): memory grows ~linearly with
    #     each batch)
    #   - https://github.com/tensorflow/tensorflow/issues/33898
    #     (tf.GradientTape training much slower than keras.fit)
    # None of those threads landed on one universally-confirmed root cause
    # inside TF's C++ runtime, so this comment does not claim to know the
    # exact internal mechanism either - but decorating train_step with
    # @tf.function is the concrete, standard fix that all of them converge
    # on, and it is exactly what Keras' own compiled path already does.
    # This also matches "計算の蓄積": each epoch, this step was being
    # re-traced/re-interpreted by Python from scratch; @tf.function traces
    # it once (per input shape) and reuses the compiled graph, so the
    # per-batch cost should stay flat instead of climbing.
    @tf.function
    def train_step(self, data):
        # Diagnostic only (temporary): a plain Python print() inside a
        # @tf.function-decorated method executes ONLY while TensorFlow is
        # tracing/building the graph, never on the replayed compiled calls
        # afterward. So if @tf.function is doing its job, this line should
        # print at most a small, fixed number of times total (one per
        # distinct input shape/dtype it sees - normally 1, or 2 if the
        # last, smaller batch of an epoch has a different shape) and then
        # stop appearing for the rest of the run. If it keeps printing on
        # every single batch/epoch instead, that means train_step is being
        # re-traced constantly and @tf.function is not actually preventing
        # the per-call Python/graph-build overhead - which would explain
        # why the previous @tf.function fix did not stop epoch time from
        # growing on Admin. Remove this line once the question is settled.
        print("[Diag] train_step is being TRACED now (should be rare, not every batch)")
        if self.supervised:
            x, y = data
        else:
            x, y = data, None

        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encoder(x, training=True)
            x_recon  = self.decoder(z, training=True)
            # Fix: pass z_mean directly to regression head (no second encoder call)
            y_pred   = self.regression(z_mean, training=True) if self.supervised else None
            total_loss, loss_recon, loss_KL, loss_prop = self.compute_loss(
                x, y, x_recon, y_pred, z_mean, z_log_var)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss":       total_loss,
                "recon_loss": loss_recon,
                "kl_loss":    loss_KL,
                "prop_loss":  loss_prop}


    # ==================== Utility methods ====================
    def compress_to_latent(self, x, verbose=0):
        """Compress input data to latent representation z."""
        z, _, _ = self.encoder.predict(x, verbose=verbose)
        return z

    def restoration_from_latent(self, z, verbose=0):
        """Reconstruct data from latent representation z."""
        return self.decoder.predict(z, verbose=verbose)

    def predict_y(self, x, verbose=0):
        """Predict property y from input x (supervised mode only)."""
        if not self.supervised or self.regression is None:
            raise RuntimeError("predict_y() is only available when supervised=True.")
        # Fix: get z_mean from encoder, then pass to regression head directly
        _, z_mean, _ = self.encoder.predict(x, verbose=verbose)
        return self.regression.predict(z_mean, verbose=verbose)


    # ==================== Training loop ====================
    def fit(self, x, y=None, batch_size=None, epochs=1, verbose=1,
            callbacks=None, shuffle=True, **kwargs):

        print(f"==================== FTCP_VAE pattern {self.network_pattern} training started ====================")
        if self.supervised:
            if isinstance(x, (tuple, list)) and len(x) == 2:
                x_data, y_data = x
            else:
                raise ValueError("fit expects x=(X_train, y_train) when supervised=True")
        else:
            x_data, y_data = x, None

        if x_data.ndim == 2:
            x_data = x_data[..., None]
        n_samples = x_data.shape[0]
        if batch_size is None:
            batch_size = min(32, n_samples)

        # Fix (correction of the earlier .cache() fix, which did NOT work):
        # the diagnostic run on Admin showed epoch time still climbing
        # (122s -> 128s -> 135s -> 153s...) even with .cache() AND
        # @tf.function on train_step, while tracemalloc/gc showed
        # essentially flat Python-level memory and the "[Diag] TRACED"
        # print appeared only at the very start of the run (no repeated
        # retracing). So neither eager-mode overhead nor a Python-level
        # leak was the cause. But the log itself keeps showing, every
        # single epoch:
        #   "Allocation of 5872013312 exceeds 10% of system memory"
        # 5872013312 bytes is the size of the whole X_train array. That
        # warning recurring on every epoch boundary - even well after
        # .cache() should have finished materializing the dataset once -
        # means the tf.data pipeline (from_tensor_slices on a huge numpy
        # array, wrapped in .cache()/.shuffle()/.batch()) keeps touching
        # the full ~5.9GB array as a whole on every epoch, at the TensorFlow
        # C++ allocator level. tracemalloc cannot see this because it only
        # tracks Python-heap allocations, not TF's own native allocator -
        # which is exactly why the previous diagnostic looked "flat" while
        # the real slowdown (most likely OS-level heap fragmentation /
        # paging from repeatedly allocating and freeing a ~5.9GB block)
        # kept building up underneath it.
        # from_tensor_slices()+.cache() is the wrong tool for data that
        # already fits fully in RAM as a single array; that pipeline is
        # meant for streaming data that does NOT already fit in memory.
        # Replacing it with plain NumPy index-based batching below never
        # touches the full array as one block after the initial load - it
        # only ever copies out one small batch (batch_size rows) at a
        # time - so it should not trigger this per-epoch allocation at all.
        if callbacks is None:
            callbacks = []

        for cb in callbacks:
            cb.set_model(self)
            cb.on_train_begin({})

        # Diagnostic only (temporary): track Python-level allocated memory
        # (tracemalloc, stdlib, works on Windows) and the live Python object
        # count (gc) once per epoch. This is to directly observe whether
        # something is actually accumulating epoch over epoch (memory leak /
        # growing object graph), rather than guessing again. Remove once the
        # question is settled.
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        for epoch in range(self.start_epoch, epochs):
            for cb in callbacks:
                cb.on_epoch_begin(epoch, {})
            print(f"\nEpoch {epoch+1}/{epochs}\nTrain on {n_samples} samples")
            start_time = time.time()
            step_count = 0

            # Plain NumPy batching (see comment above): shuffle indices, then
            # slice out one small batch of rows at a time. This never
            # materializes/copies the full n_samples-row array as a whole -
            # only batch_size rows are copied per step - so it should not
            # reproduce the "Allocation of 5872013312 exceeds 10%" warning
            # or whatever was behind it.
            perm = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)

            for step, start in enumerate(range(0, n_samples, batch_size)):
                idx = perm[start:start + batch_size]
                x_batch = x_data[idx]
                if self.supervised:
                    batch = (x_batch, y_data[idx])
                else:
                    batch = x_batch

                step_count += 1
                logs        = self.train_step(batch)
                logs_scalar = {k: float(v.numpy()) for k, v in logs.items()}
                for cb in callbacks:
                    cb.on_train_batch_end(step, logs_scalar)
                done    = min(step_count * batch_size, n_samples)
                elapsed = time.time() - start_time
                bar_len = 30
                percent = int(bar_len * done / n_samples)
                bar     = "=" * percent + ">" + "." * (bar_len - percent - 1)
                sys.stdout.write(
                    f"\r{done}/{n_samples} [{bar}] - {elapsed:4.0f}s elapsed - loss: {logs_scalar['loss']:.4f}"
                )
                sys.stdout.flush()

            print(f"\nEpoch {epoch+1}: total={logs_scalar['loss']:.6f}, "
                  f"recon={logs_scalar['recon_loss']:.6f}, "
                  f"KL={logs_scalar['kl_loss']:.6f}, "
                  f"prop={logs_scalar['prop_loss']:.6f}")

            # Diagnostic only (temporary): see comment near tracemalloc.start()
            # above. gc.collect() is called first so the object count reflects
            # genuinely-unreachable-but-not-yet-freed garbage too (a classic
            # symptom of reference cycles that the generational GC struggles
            # to reclaim promptly in long eager-mode loops).
            gc.collect()
            _current_mem, _peak_mem = tracemalloc.get_traced_memory()
            print(
                f"[Diag] epoch {epoch+1}: elapsed={elapsed:.1f}s, "
                f"tracemalloc current={_current_mem/1e6:.1f}MB peak={_peak_mem/1e6:.1f}MB, "
                f"gc_objects={len(gc.get_objects())}"
            )

            self.history_rows.append({
                "epoch":      epoch + 1,
                "loss_total": logs_scalar["loss"],
                "loss_recon": logs_scalar["recon_loss"],
                "loss_KL":    logs_scalar["kl_loss"],
                "loss_prop":  logs_scalar["prop_loss"]
            })
            if self.csv_path:
                df = pd.DataFrame(self.history_rows)
                df.to_csv(self.csv_path, index=False, float_format="%.7f")

            for cb in callbacks:
                cb.on_epoch_end(epoch, logs_scalar)

            if self.vae_path:
                self.save_weights(self.vae_path)
                print(f"[Info] Model weights saved to {self.vae_path}")

        for cb in callbacks:
            cb.on_train_end({})
        return None


    # ==================== compile ====================
    def compile(self, optimizer=None, loss=None, **kwargs):
        """
        Uses the optimizer passed from the caller.
        loss argument is a dummy required for Keras API compatibility.
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
        def dummy_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred))
        super().compile(optimizer=optimizer, loss=dummy_loss, **kwargs)