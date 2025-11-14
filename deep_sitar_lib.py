# deep_sitar_lib.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from time import time

# ============================================================
#  CONFIGURATION STRUCTURES
# ============================================================

@dataclass
class Paths:
    base: Path
    data: Path
    plots: Path
    results: Path

    @staticmethod
    def from_base(base: Path) -> "Paths":
        data = base / "data"
        plots = base / "plots"
        results = base / "results"
        for d in (data, plots, results):
            d.mkdir(parents=True, exist_ok=True)
        return Paths(base, data, plots, results)


@dataclass
class TrainConfig:
    data_s: int
    train_frac: float = 0.8
    epp: list[int] = None
    lr: list[float] = None
    batch_size: int = 100
    num_seg: int = 10
    degree: int = 3
    lam_smooth: float = 1e-4
    use_intercept: bool = False
    natural: bool = False
    n_obs: int | None = None  # <---- NEW: number of time points per subject

    def __post_init__(self):
        if self.epp is None:
            self.epp = [5000, 8000, 5000]
        if self.lr is None:
            self.lr = [1e-2, 1e-4, 1e-5]

# ============================================================
#  UTILITIES
# ============================================================

AUTO_CLOSE_SECS = 10

def save_and_show(fig, path: Path, dpi=400, close_secs=AUTO_CLOSE_SECS):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    except Exception as e:
        print(f"[WARN] Could not save {path}: {e}")
    try:
        plt.show(block=False)
        plt.pause(close_secs)
    finally:
        plt.close(fig)


def plot_comparison(X_exact, Y_exact, Y_estimation, title=None, save_path: Path | None = None):
    fig, ax = plt.subplots()
    X_np = np.asarray(X_exact).ravel()
    Y_np = np.asarray(Y_exact).ravel()
    Yhat_np = np.asarray(Y_estimation).ravel()
    ax.scatter(X_np, Y_np, marker='o', alpha=0.2, label='Exact')
    ax.scatter(X_np, Yhat_np, marker='>', alpha=1.0, label='Estimation')
    if title:
        ax.set_title(title)
    ax.set_ylabel("Y")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=2)
    if save_path is not None:
        save_and_show(fig, Path(save_path))
    else:
        plt.show()


def plot_loss(his_loss_train, his_loss_val, chain, lr_list, n_id, obs, num_seg, save_path: Path | None = None):
    fig, ax = plt.subplots()
    train_arr = np.asarray(his_loss_train).reshape(-1)
    val_arr = np.asarray(his_loss_val).reshape(-1)
    if chain <= 0 or chain > len(train_arr):
        chain = len(train_arr)
    ax.semilogy(train_arr[-chain:], label='Training Loss')
    ax.semilogy(val_arr[-chain:], label='Validation Loss')
    ax.set_title(f'n_id={n_id} Obs={obs} LR={lr_list}')
    ax.set_ylabel("log-Loss")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=2)
    if save_path is not None:
        save_and_show(fig, Path(save_path))
    else:
        plt.show()

# ============================================================
#  DATA LOADING AND PREPROCESSING
# ============================================================

def make_tensors_from_df(df: pd.DataFrame, data_s: int, train_frac: float, n_obs: int | None = None):
    # infer n_obs automatically
    if n_obs is None:
        n_obs = int(df.shape[0] / data_s)

    # reshape
    X_raw = df["age"].values.reshape((data_s, n_obs))
    Y_raw = df["y"].values.reshape((data_s, n_obs))
    if "height" in df.columns:
        f_raw = df["height"].values.reshape((data_s, n_obs))
    else:
        f_raw = np.zeros_like(Y_raw)

    # centers
    X_mean = np.mean(X_raw)
    Y_mean = np.mean(Y_raw)
    f_mean = np.mean(f_raw)

    X_center = X_raw - X_mean
    Y_center = Y_raw - Y_mean
    f_center = f_raw - f_mean

    # split
    t_s = int(data_s * train_frac)
    X_training = tf.convert_to_tensor(X_center[:t_s, :], dtype=tf.float32)
    Y_training = tf.convert_to_tensor(Y_center[:t_s, :], dtype=tf.float32)
    f_training = tf.convert_to_tensor(f_center[:t_s, :], dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_center[t_s:, :], dtype=tf.float32)
    Y_val = tf.convert_to_tensor(Y_center[t_s:, :], dtype=tf.float32)
    f_val = tf.convert_to_tensor(f_center[t_s:, :], dtype=tf.float32)

    # reference X
    XX = df["age"].values.reshape((data_s, n_obs))[0:1, :].reshape([1, n_obs])
    X_ref = tf.convert_to_tensor(XX - np.mean(XX), dtype=tf.float32)

    # domain
    xmin = float(tf.reduce_min(X_training))
    xmax = float(tf.reduce_max(X_training))
    pad = 1e-3 * (xmax - xmin)
    domain = (xmin - pad, xmax + pad)

    return (X_training, Y_training, f_training,
            X_val, Y_val, f_val,
            X_ref, (X_mean, Y_mean, f_mean), domain, n_obs)

# ============================================================
#  BSPLINE LAYER
# ============================================================

class BSplineLayer(tf.keras.layers.Layer):
    def __init__(self, num_seg, degree, domain, use_intercept=True, natural=False, **kwargs):
        super().__init__(**kwargs)
        self.num_seg = int(num_seg)
        self.degree = int(degree)
        self.domain = tuple(domain)
        self.use_intercept = use_intercept
        self.natural = natural
        if self.natural and self.degree < 2:
            raise ValueError("Natural spline requires degree >= 2.")
        self.num_bases = self.num_seg + self.degree
        span = self.domain[1] - self.domain[0]
        dx = span / self.num_seg
        self.knots = np.arange(
            self.domain[0] - self.degree * dx - dx / 2,
            self.domain[1] + self.degree * dx + dx / 2,
            dx, dtype=np.float32
        )
        if self.natural:
            C = self._build_C_second_deriv_matrix(self.domain[0], self.domain[1])
            R = self._null_space(C)
            self.R = tf.constant(R.astype(np.float32))
            self.control_points_reduced = tf.Variable(
                initial_value=tf.random.normal([self.num_bases - 2, 1], stddev=0.1),
                trainable=True, name="control_points_reduced")
        else:
            self.control_points = tf.Variable(
                initial_value=tf.random.normal([self.num_bases, 1], stddev=0.1),
                trainable=True, name="control_points")

    def bspline_basis_tensor(self, x, i, k):
        if k == 0:
            return tf.cast((self.knots[i] <= x) & (x < self.knots[i + 1]), tf.float32)
        denom1 = self.knots[i + k] - self.knots[i] + 1e-8
        denom2 = self.knots[i + k + 1] - self.knots[i + 1] + 1e-8
        term1 = tf.where(
            denom1 > 0,
            (x - self.knots[i]) / denom1 * self.bspline_basis_tensor(x, i, k - 1), 0.0)
        term2 = tf.where(
            denom2 > 0,
            (self.knots[i + k + 1] - x) / denom2 * self.bspline_basis_tensor(x, i + 1, k - 1), 0.0)
        return term1 + term2

    def _basis_deriv_scalar(self, x, i, k, d):
        if d == 0: return self._basis_scalar(x, i, k)
        if k == 0: return 0.0
        denom1 = float(self.knots[i + k] - self.knots[i]) + 1e-8
        denom2 = float(self.knots[i + k + 1] - self.knots[i + 1]) + 1e-8
        return (k / denom1 * self._basis_deriv_scalar(x, i, k - 1, d - 1)
                - k / denom2 * self._basis_deriv_scalar(x, i + 1, k - 1, d - 1))

    def _basis_scalar(self, x, i, k):
        if k == 0:
            return 1.0 if (self.knots[i] <= x < self.knots[i + 1]) else 0.0
        denom1 = float(self.knots[i + k] - self.knots[i]) + 1e-8
        denom2 = float(self.knots[i + k + 1] - self.knots[i + 1]) + 1e-8
        t1 = ((x - float(self.knots[i])) / denom1) * self._basis_scalar(x, i, k - 1) if denom1 > 0 else 0.0
        t2 = ((float(self.knots[i + k + 1]) - x) / denom2) * self._basis_scalar(x, i + 1, k - 1) if denom2 > 0 else 0.0
        return t1 + t2

    def _build_C_second_deriv_matrix(self, a, b):
        C = np.zeros((2, self.num_bases), dtype=np.float64)
        for i in range(self.num_bases):
            C[0, i] = self._basis_deriv_scalar(a, i, self.degree, d=2)
            b_left = np.nextafter(b, -np.inf)
            C[1, i] = self._basis_deriv_scalar(b_left, i, self.degree, d=2)
        return C

    def _null_space(self, C, rtol=1e-10):
        U, S, Vh = np.linalg.svd(C, full_matrices=True)
        tol = rtol * max(C.shape) * (S[0] if S.size > 0 else 1.0)
        rank = (S > tol).sum()
        R = Vh[rank:].T
        return R

    def call(self, inputs):
        if len(inputs.shape) == 1:
            inputs = tf.expand_dims(inputs, axis=-1)
        splines = [self.bspline_basis_tensor(inputs, i, self.degree) for i in range(self.num_bases)]
        splines = tf.stack(splines, axis=-1)
        cp_full = (tf.matmul(self.R, self.control_points_reduced)
                   if self.natural else self.control_points)
        if not self.use_intercept:
            splines = splines[:, :, 1:]; cp = cp_full[1:]
        else:
            cp = cp_full
        weighted = tf.tensordot(splines, cp, axes=[[2], [0]])
        return tf.squeeze(weighted, axis=-1)

# ============================================================
#  DEEP-SITAR NETWORK
# ============================================================

def build_encoder(inp_dim: int, hidden: list[int], out_dim: int, acts=('tanh', 'tanh', 'linear'), kernel_init=None):
    if kernel_init is None:
        kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
    model = Sequential(name="DeepRandEffects")
    for i, (units, act) in enumerate(zip(hidden + [out_dim], acts)):
        if i == 0:
            model.add(Dense(units, input_shape=(inp_dim,), activation=act, kernel_initializer=kernel_init, use_bias=True))
        else:
            model.add(Dense(units, activation=act, kernel_initializer=kernel_init, use_bias=True))
    return model


class DeepSITAR(tf.keras.Model):
    def __init__(self, x_ref, spline_layer, deep_model, name=None):
        super().__init__(name=name)
        self.loc_net_S = spline_layer
        self.loc_net_deep = deep_model
        self.X = x_ref

    def call(self, x):
        rand_effects = self.loc_net_deep(x)
        dim_x = tf.shape(self.X)[1]
        a_i = tf.tile(rand_effects[:, 0:1], (1, dim_x))
        b_i = tf.tile(rand_effects[:, 1:2], (1, dim_x))
        c_i = tf.tile(rand_effects[:, 2:3], (1, dim_x))
        out_spl = self.loc_net_S((self.X - b_i) * tf.math.exp(c_i))
        return a_i + out_spl

    def MyELBO(self, y_true, y_pred):
        re = self.loc_net_deep(y_true)
        re_center = re - tf.reduce_mean(re, axis=0, keepdims=True)
        cov = tfp.stats.covariance(re_center)
        cov = cov + 1e-6 * tf.eye(tf.shape(cov)[0])
        precision = tf.linalg.pinv(cov)
        mat1 = tf.linalg.matmul(re_center, precision)
        mat2 = tf.linalg.matmul(mat1, re_center, transpose_b=True)
        penal_diag = tf.linalg.diag_part(mat2)
        penal = tf.reshape(penal_diag, (-1, 1))
        recon = tf.reduce_mean((y_pred - y_true) ** 2)
        reg = tf.reduce_mean(penal)
        return recon + reg

    def ret_fix(self, x):
        return tf.reduce_mean(self.loc_net_deep(x), axis=0).numpy()

    def ret_sig(self, x):
        return tfp.stats.covariance(self.loc_net_deep(x)).numpy()

# ============================================================
#  TRAINING FUNCTION
# ============================================================

def train_deep_sitar(X_training, Y_training, X_val, Y_val, x_ref, domain, cfg: TrainConfig, paths: Paths, seed: int = 420):
    np.random.seed(seed); tf.random.set_seed(seed)
    n_obs = X_training.shape[1]
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
    spline = BSplineLayer(cfg.num_seg, cfg.degree, domain, cfg.use_intercept, cfg.natural)
    enc = build_encoder(inp_dim=n_obs, hidden=[n_obs, n_obs], out_dim=3,
                        acts=('tanh', 'tanh', 'linear'), kernel_init=initializer)
    model = DeepSITAR(x_ref, spline, enc)
    his_train, his_val = [], []
    start = time()
    for lr in cfg.lr:
        opt = Adam(learning_rate=lr, epsilon=1e-7)
        model.compile(optimizer=opt, loss=model.MyELBO)
        hist = model.fit(Y_training, Y_training, epochs=cfg.epp[cfg.lr.index(lr)],
                         verbose=1, batch_size=cfg.batch_size,
                         validation_data=(Y_val, Y_val))
        his_train.extend(hist.history['loss'])
        his_val.extend(hist.history['val_loss'])
    train_minutes = (time() - start) / 60
    n_id = X_training.shape[0] + X_val.shape[0]
    plot_loss(his_train, his_val, 0, cfg.lr, n_id, n_obs, cfg.num_seg, paths.plots / f"loss_full_{cfg.num_seg}.pdf")
    return model, his_train, his_val, train_minutes

# ============================================================
#  EXPORT RESULTS (add this to deep_sitar_lib.py)
# ============================================================
# ============================================================
#  EXPORT RESULTS AND PLOTS
# ============================================================

def export_all(
    model,
    paths,
    cfg,
    means_tuple,
    X_training, Y_training, f_training,
    X_val, Y_val, f_val,
    his_loss_train, his_loss_val,
    train_minutes: float,
):
    """
    Save plots, training losses, fixed and random effects, and fitted curves.

    Parameters
    ----------
    model : DeepSITAR
        Trained model.
    paths : Paths
        Folder structure (plots, results).
    cfg : TrainConfig
        Model configuration.
    means_tuple : tuple
        (X_mean, Y_mean, f_mean) for de-centering outputs.
    *_training, *_val : tensors
        Data used in training and validation.
    his_loss_train, his_loss_val : list
        Loss histories.
    train_minutes : float
        Total training time in minutes.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from pathlib import Path

    from deep_sitar_lib import plot_comparison, save_and_show

    X_mean, Y_mean, f_mean = means_tuple
    n_id = int(X_training.shape[0] + X_val.shape[0])

    # ========================================================
    # 1️⃣ Plot comparison: Validation true vs predicted
    # ========================================================
    plot_comparison(
        X_val,
        f_val,
        model(Y_val),
        title=f"Validation: f vs estimation (n_ind={n_id}, seg={cfg.num_seg})",
        save_path=paths.plots / f'compare_f_vs_est_seg{cfg.num_seg}.pdf'
    )

    # ========================================================
    # 2️⃣ Scatter plot f vs ŷ (identity line)
    # ========================================================
    fig, ax = plt.subplots(figsize=(6, 6))
    yhat_val = model(Y_val).numpy()
    f_val_np = f_val.numpy()
    ax.scatter(f_val_np.ravel(), yhat_val.ravel(), alpha=0.5, label='Predicted')
    minv = float(min(np.min(f_val_np), np.min(yhat_val)))
    maxv = float(max(np.max(f_val_np), np.max(yhat_val)))
    ax.plot([minv, maxv], [minv, maxv], 'r--', lw=1.5, label='Identity line')
    ax.set_xlabel("True f (validation)")
    ax.set_ylabel("Predicted f̂")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_and_show(fig, paths.plots / f'scatter_val_seg{cfg.num_seg}.pdf')

    # ========================================================
    # 3️⃣ Plot training and validation loss evolution
    # ========================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(his_loss_train, label='Training Loss')
    ax.semilogy(his_loss_val, label='Validation Loss')
    ax.set_title(f'Loss evolution (segments={cfg.num_seg})')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('log-Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_and_show(fig, paths.plots / f'loss_history_seg{cfg.num_seg}.pdf')

    # ========================================================
    # 4️⃣ Fixed & covariance (random effects)
    # ========================================================
    fixed_est_values = model.ret_fix(Y_training)
    sigma_est_values = model.ret_sig(Y_training)

    # Save numerical info to .txt for easy reading
    summary_txt = paths.results / f'summary_seg{cfg.num_seg}.txt'
    with open(summary_txt, "w") as f:
        f.write(f"Deep-SITAR summary (num_seg={cfg.num_seg})\n")
        f.write(f"Training time (min): {train_minutes:.3f}\n\n")
        f.write(f"Fixed effects (mean of random effects):\n{fixed_est_values}\n\n")
        f.write(f"Covariance matrix of random effects:\n{sigma_est_values}\n")

    print(f"✅ Summary written to: {summary_txt}")

    # ========================================================
    # 5️⃣ Export CSVs for further analysis
    # ========================================================
    X_train_np = X_training.numpy().ravel()
    Y_train_np = Y_training.numpy().ravel()
    f_train_np = f_training.numpy().ravel()
    y_train_pred_np = model(Y_training).numpy().ravel()

    X_val_np = X_val.numpy().ravel()
    Y_val_np = Y_val.numpy().ravel()
    f_val_np = f_val.numpy().ravel()
    y_val_pred_np = model(Y_val).numpy().ravel()

    # Training data CSV
    df_train = pd.DataFrame({
        'x': X_train_np + X_mean,
        'y': Y_train_np + Y_mean,
        'f': f_train_np + f_mean,
        'y_pred': y_train_pred_np + Y_mean,
        'num_seg': cfg.num_seg,
        'nind': n_id,
        'type': 'train'
    })

    # Validation data CSV
    df_val = pd.DataFrame({
        'x': X_val_np + X_mean,
        'y': Y_val_np + Y_mean,
        'f': f_val_np + f_mean,
        'y_pred': y_val_pred_np + Y_mean,
        'num_seg': cfg.num_seg,
        'nind': n_id,
        'type': 'val'
    })

    # Loss CSV
    df_loss = pd.DataFrame({
        'epoch': np.arange(len(his_loss_train)),
        'loss_train': his_loss_train,
        'loss_val': his_loss_val,
        'num_seg': cfg.num_seg,
        'nind': n_id,
        'time_min': train_minutes,
    })

    # Fixed and covariance CSVs
    df_fixed = pd.DataFrame({'fixed_est': fixed_est_values})
    df_cov = pd.DataFrame(sigma_est_values)

    # ========================================================
    # 5️⃣ Export CSVs for further analysis (filename includes num_seg + n_id)
    # ========================================================

    # Compose dynamic filename suffix
    suffix = f"_seg{cfg.num_seg}_nind{n_id}"

    # Save all CSVs with descriptive filenames
    df_train.to_csv(paths.results / f'deepSITAR_train{suffix}.csv', index=False)
    df_val.to_csv(paths.results / f'deepSITAR_val{suffix}.csv', index=False)
    df_loss.to_csv(paths.results / f'deepSITAR_loss{suffix}.csv', index=False)
    df_fixed.to_csv(paths.results / f'deepSITAR_fixed{suffix}.csv', index=False)
    df_cov.to_csv(paths.results / f'deepSITAR_cov{suffix}.csv', index=False)

    print(f"✅ CSV files saved in: {paths.results}")
    print(f"   (suffix used: {suffix})")
    # ========================================================
    # 6️⃣ Return key info
    # ========================================================
    return {
        "fixed_est": fixed_est_values,
        "cov_est": sigma_est_values,
        "train_minutes": train_minutes,
        "plots_dir": paths.plots,
        "results_dir": paths.results
    }
