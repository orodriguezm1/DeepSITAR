# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from pathlib import Path

# ================================
#  Configuración y utilidades
# ================================

# Creación segura de carpetas
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"
for d in (DATA_DIR, PLOTS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Semillas (reproducibilidad)
np.random.seed(420)
tf.random.set_seed(420)

# Auto-cierre de figuras
AUTO_CLOSE_SECS = 10

def save_and_show(fig, path: Path, dpi=400, close_secs=AUTO_CLOSE_SECS):
    """Guarda la figura y la muestra; se cierra sola pasados close_secs segundos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    except Exception as e:
        print(f"[WARN] No se pudo guardar {path}: {e}")
    try:
        plt.show(block=False)
        plt.pause(close_secs)
    finally:
        plt.close(fig)

# ================================
#  Parámetros del experimento
# ================================
start_time = time()
data_s = 500  # 500, 1000 o 5000
train_frac = 0.8
t_s = int(data_s * train_frac)

# Entrenamiento multietapa
#epp = [8000, 10000]  # épocas por fase
#lr = [1e-3, 1e-4]   # learning rates por fase
#b_size = 100        # batch size

epp = [5000,8000, 5000] # Épocas para cada fase de entrenamiento
#epp = [5000,5000, 5000] # Épocas para cada fase de entrenamiento
lr =  [1e-2, 1e-4, 1e-5]     # Tasas de aprendizaje para cada fase
b_size = 100          # Tamaño del batch


# ================================
#  Carga y preparación de datos
# ================================
valid_sizes = {500, 1000, 5000}
if data_s not in valid_sizes:
    raise ValueError(f"data_s debe ser uno de {sorted(valid_sizes)}, recibido: {data_s}")

csv_path = DATA_DIR / f"simul_data_{data_s}_20new.csv"
if not csv_path.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo esperado: {csv_path}\n"
        f"Colócalo en la carpeta 'data' o ajusta el nombre."
    )

df = pd.read_csv(csv_path)

# Validaciones básicas de columnas
for col in ("age", "y", "height"):
    if col not in df.columns:
        raise KeyError(f"Falta la columna '{col}' en el CSV.")

# Reshape a (data_s, 20)
try:
    X_raw = df["age"].values.reshape((data_s, 20))
    Y_raw = df["y"].values.reshape((data_s, 20))
    f_raw = df["height"].values.reshape((data_s, 20))
except Exception as e:
    raise ValueError(
        "No fue posible reshapedar a (data_s, 20). "
        "Verifica que el CSV tenga exactamente data_s*20 filas por cada variable."
    ) from e

# Centrados (manteniendo tu tratamiento actual)
X_mean = np.mean(X_raw)
Y_mean = np.mean(Y_raw)
f_mean = np.mean(f_raw)

X_center = X_raw - X_mean
Y_center = Y_raw - Y_mean
f_center = f_raw - f_mean

# Tensores
X_training = tf.convert_to_tensor(X_center[:t_s, :], dtype=tf.float32)
Y_training = tf.convert_to_tensor(Y_center[:t_s, :], dtype=tf.float32)
f_training = tf.convert_to_tensor(f_center[:t_s, :], dtype=tf.float32)

X_val = tf.convert_to_tensor(X_center[t_s:, :], dtype=tf.float32)
Y_val = tf.convert_to_tensor(Y_center[t_s:, :], dtype=tf.float32)
f_val = tf.convert_to_tensor(f_center[t_s:, :], dtype=tf.float32)

# Un ejemplo de X de una sola fila para inicialización del modelo
XX = df["age"].values.reshape((data_s, 20))[0:1, :].reshape([1, 20])
X = tf.convert_to_tensor(XX - np.mean(XX), dtype=tf.float32)

# Dominio para la B-spline basado en X_training (con pequeño padding)
xmin = float(tf.reduce_min(X_training).numpy())
xmax = float(tf.reduce_max(X_training).numpy())
pad = 1e-3 * max(1.0, abs(xmax - xmin))
domain = (xmin - pad, xmax + pad)

initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)

# ================================
#  Funciones de visualización
# ================================
def plot_comparison(X_exact, Y_exact, Y_estimation, title=None, save_path=None):
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
    if save_path is None:
        save_path = PLOTS_DIR / "plot_comparison.pdf"
    save_and_show(fig, Path(save_path))

def plot_loss(his_loss_train, his_loss_val, chain, lr_list, n_id=500, obs=20, num_seg=5):
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
    out_path = PLOTS_DIR / f'plot_estim_chain_{chain}_{num_seg}.pdf'
    save_and_show(fig, out_path)

# ================================
#  Capa B-Spline
# ================================
# ================================
#  Capa B-Spline (con opción de intercepto y natural spline)
# ================================
import numpy as np
import tensorflow as tf

class BSplineLayer(tf.keras.layers.Layer):
    """
    B-spline con opción de 'natural=True' imponiendo f''(a)=0 y f''(b)=0.
    Mantiene las bases B-spline (Cox–de Boor). La naturalidad se impone
    parametrizando los coeficientes c = R @ theta, donde las columnas de R
    generan el núcleo de C (C c = 0), con C_{j,i} = B_i''(x_j), j=1,2 (extremos).
    """
    def __init__(self, num_seg, degree, domain, use_intercept=True, natural=False, **kwargs):
        super().__init__(**kwargs)
        self.num_seg = int(num_seg)
        self.degree = int(degree)
        self.domain = tuple(domain)
        self.use_intercept = use_intercept
        self.natural = natural
        if self.natural and self.degree < 2:
            raise ValueError("Natural spline requiere degree >= 2 (segunda derivada).")

        # nº de funciones base (sin aplicar restricciones/intercepto)
        self.num_bases = self.num_seg + self.degree

        # Nudos (uniformes extendidos como en tu código)
        span = (self.domain[1] - self.domain[0])
        dx = span / self.num_seg

        self.knots = np.arange(
            self.domain[0] - self.degree * dx- dx / 2,
            self.domain[1] + self.degree * dx + dx / 2,  # +dx/2 para incluir el último nudo
            dx,
            dtype=np.float32
        )

        # Variables entrenables: si natural -> parámetros reducidos; si no, coeficientes completos
        if self.natural:
            # R se construye en __init__ (en CPU) y se congela como constante
            C = self._build_C_second_deriv_matrix(self.domain[0], self.domain[1])
            R = self._null_space(C)  # [num_bases, num_bases-2]
            if R.shape[1] != self.num_bases - 2:
                # En casos degenerados numéricamente, caemos a QR para completar base
                R = self._complete_basis_from_null(C)

            # Chequeo: C @ R debe ser ~0 si las restricciones se cumplen
            print("Chequeo restricción natural:")
            print(C @ R)

            self.R = tf.constant(R.astype(np.float32), dtype=tf.float32)
            self.control_points_reduced = tf.Variable(
                initial_value=tf.random.normal([self.num_bases - 2, 1], stddev=0.1),
                trainable=True, name="control_points_reduced"
            )
        else:
            self.control_points = tf.Variable(
                initial_value=tf.random.normal([self.num_bases, 1], stddev=0.1),
                trainable=True, name="control_points"
            )

    # ---------- Construcción base B-spline (Cox–de Boor) ----------
    def bspline_basis_tensor(self, x, i, k):
        """Base B-spline B_{i,k}(x) con recursión de Cox–de Boor (vectorizada en x)."""
        if k == 0:
            return tf.cast((self.knots[i] <= x) & (x < self.knots[i + 1]), tf.float32)
        denom1 = self.knots[i + k] - self.knots[i] + 1e-8
        denom2 = self.knots[i + k + 1] - self.knots[i + 1] + 1e-8
        term1 = tf.where(
            denom1 > 0,
            (x - self.knots[i]) / denom1 * self.bspline_basis_tensor(x, i, k - 1),
            0.0
        )
        term2 = tf.where(
            denom2 > 0,
            (self.knots[i + k + 1] - x) / denom2 * self.bspline_basis_tensor(x, i + 1, k - 1),
            0.0
        )
        return term1 + term2

    # ---------- Evaluación escalar (para derivadas en extremos) ----------
    def _basis_scalar(self, x, i, k):
        if k == 0:
            return 1.0 if (self.knots[i] <= x < self.knots[i + 1]) else 0.0
        denom1 = float(self.knots[i + k] - self.knots[i]) + 1e-8
        denom2 = float(self.knots[i + k + 1] - self.knots[i + 1]) + 1e-8
        t1 = ((x - float(self.knots[i])) / denom1) * self._basis_scalar(x, i, k - 1) if denom1 > 0 else 0.0
        t2 = ((float(self.knots[i + k + 1]) - x) / denom2) * self._basis_scalar(x, i + 1, k - 1) if denom2 > 0 else 0.0
        return t1 + t2

    def _basis_deriv_scalar(self, x, i, k, d):
        """
        d-ésima derivada de B_{i,k} en x, escalar (para construir C).
        Fórmula recursiva: B'_{i,k} = k/(t_{i+k}-t_i) B_{i,k-1} - k/(t_{i+k+1}-t_{i+1}) B_{i+1,k-1}
        y para orden d>1, se aplica recursivamente.
        """
        if d == 0:
            return self._basis_scalar(x, i, k)
        if k == 0:
            return 0.0
        denom1 = float(self.knots[i + k] - self.knots[i]) + 1e-8
        denom2 = float(self.knots[i + k + 1] - self.knots[i + 1]) + 1e-8
        return (
            k / denom1 * self._basis_deriv_scalar(x, i, k - 1, d - 1)
            - k / denom2 * self._basis_deriv_scalar(x, i + 1, k - 1, d - 1)
        )

    # ---------- Matriz de restricciones y su núcleo ----------
    def _build_C_second_deriv_matrix(self, a, b):
        """Construye C de tamaño 2 x num_bases con C[j,i] = B_i''(x_j), x_1=a, x_2=b."""
        C = np.zeros((2, self.num_bases), dtype=np.float64)
        for i in range(self.num_bases):
            C[0, i] = self._basis_deriv_scalar(a, i, self.degree, d=2)
            # Para el extremo derecho, tratar el punto final como límite por la izquierda:
            b_left = np.nextafter(b, -np.inf)
            C[1, i] = self._basis_deriv_scalar(b_left, i, self.degree, d=2)
        return C

    def _null_space(self, C, rtol=1e-10):
        """
        Núcleo de C por SVD: devuelve matriz R (num_bases x m) cuyas columnas
        forman una base del núcleo. m = num_bases - rank(C) (idealmente 2).
        """
        U, S, Vh = np.linalg.svd(C, full_matrices=True)
        tol = rtol * max(C.shape) * (S[0] if S.size > 0 else 1.0)
        null_mask = (S <= tol)
        # Si rank=2, null tiene dimensión num_bases-2 -> tomamos últimas columnas de Vh
        if null_mask.size >= 2:
            rank = np.sum(~null_mask)
        else:
            rank = np.linalg.matrix_rank(C)
        R = Vh[rank:].T  # columnas correspondientes al núcleo
        return R

    def _complete_basis_from_null(self, C):
        """
        Fallback robusto: si por numérica el SVD no dio exactamente dim núc = num_bases-2.
        Usamos SVD pero forzamos a coger las últimas (num_bases-2) columnas de Vh.
        """
        U, S, Vh = np.linalg.svd(C, full_matrices=True)
        want = self.num_bases - 2
        R = Vh[-want:].T  # [num_bases, want]
        return R

    # ---------- Forward ----------
    def call(self, inputs):
        # Asegurar entrada 2D: [batch, features]
        if len(inputs.shape) == 1:
            inputs = tf.expand_dims(inputs, axis=-1)

        # Construir tensor de bases: [batch, features, num_bases]
        splines = []
        for i in range(self.num_bases):
            splines.append(self.bspline_basis_tensor(inputs, i, self.degree))
        splines = tf.stack(splines, axis=-1)

        # Coeficientes efectivos
        if self.natural:
            cp_full = tf.matmul(self.R, self.control_points_reduced)  # [num_bases, 1]
        else:
            cp_full = self.control_points  # [num_bases, 1]

        # Intercepto: si no se usa, quitamos primera base en ambos lados
        if not self.use_intercept:
            splines = splines[:, :, 1:]
            cp = cp_full[1:]
        else:
            cp = cp_full

        weighted = tf.tensordot(splines, cp, axes=[[2], [0]])  # [batch, features, 1]
        return tf.squeeze(weighted, axis=-1)  # [batch, features]


# ================================
#  Instancias de modelos
# ================================
num_seg = 10 # nodos
degree = 3     # grado B-Spline

bspline_layer = BSplineLayer(num_seg=num_seg, degree=degree, domain=domain, use_intercept=False)

# --- Snapshot de pesos iniciales aleatorios (solo para plot comparativo inicial) ---
init_cp = bspline_layer.control_points.numpy().copy()

# ================================
#  PRE-ENTRENAMIENTO DEL SPLINE (LS + suavidad Δ²) Y USO COMO INICIAL
#  (con los datos centrados, tal cual tu flujo)
# ================================
x_vec = X_training.numpy().ravel().astype(np.float32)
x_vec_tensor = tf.constant(x_vec, dtype=tf.float32)

basis_count = bspline_layer.num_bases
B_cols = []
for i in range(basis_count):
    bi = bspline_layer.bspline_basis_tensor(x_vec_tensor, i, degree).numpy()
    B_cols.append(bi)
B = np.stack(B_cols, axis=1).astype(np.float32)                # [N, nb]

y_vec = Y_training.numpy().ravel().astype(np.float32)[:, None] # [N, 1]
nb = B.shape[1]

# Matriz de suavidad (Δ²)
D = np.zeros((nb - 2, nb), dtype=np.float32)
for k in range(nb - 2):
    D[k, k:k+3] = (1.0, -2.0, 1.0)

lam = 1e-4#1e-3, 1e-4  # fuerza de suavidad (ajustable)

if bspline_layer.natural:
    R = bspline_layer.R.numpy()            # [nb, nb-2]
    BR = B @ R                              # [N, nb-2]
    DR = D @ R                              # [(nb-2), nb-2]
    A = BR.T @ BR + lam * (DR.T @ DR) + 1e-6*np.eye(R.shape[1], dtype=np.float32)
    c_rhs = BR.T @ y_vec
    theta = np.linalg.solve(A, c_rhs).astype(np.float32)  # [(nb-2),1]
    cp_init = (R @ theta).astype(np.float32)              # [nb,1]
    bspline_layer.control_points_reduced.assign(theta)    # guarda θ
else:
    A = B.T @ B + lam * (D.T @ D) + 1e-6*np.eye(nb, dtype=np.float32)
    c_rhs = B.T @ y_vec
    cp_init = np.linalg.solve(A, c_rhs).astype(np.float32)
    bspline_layer.control_points.assign(cp_init)

bspline_layer.control_points.assign(cp_init)

# Ploteo de la curva B-spline pre-entrenada
x_values_pre = np.linspace(domain[0], domain[1], 400).astype(np.float32)
y_pre = bspline_layer(tf.constant(x_values_pre, dtype=tf.float32)).numpy()
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(basis_count):
    basis_y = bspline_layer.bspline_basis_tensor(tf.constant(x_values_pre, dtype=tf.float32), i, degree).numpy()
    ax.plot(x_values_pre, basis_y, linestyle='--', alpha=0.5)
ax.plot(x_values_pre, y_pre, linewidth=2, label='Pretrained B-spline Curve')
ax.set_title('Pretrained B-spline Curve and Basis Functions')
ax.legend(); ax.grid(True)
save_and_show(fig, PLOTS_DIR / f'pretrained_bspline_curve_{num_seg}.pdf')

# ================================
#  Primera estimación spline vs datos (con cp pre-entrenados)
# ================================
Spline_est = bspline_layer(X_training).numpy()
plot_comparison(
    X_training, Y_training, Spline_est,
    title="Comparación inicial Spline (pre-entrenado)",
    save_path=PLOTS_DIR / f'plot_spline_init_{num_seg}.pdf'
)

# Red neuronal (random effects a,b,c)
inp_deep = 20
out_deep = 3
nodes_deep = [20,20, out_deep]
acts = ['tanh', 'tanh', 'linear']

model_deep = Sequential(name="DeepRandEffects")
for i, (units, act) in enumerate(zip(nodes_deep, acts)):
    if i == 0:
        model_deep.add(Dense(units, input_shape=(inp_deep,),
                             activation=act, kernel_initializer=initializer,
                             use_bias=True, name="Input_layer"))
    elif i == len(nodes_deep) - 1:
        model_deep.add(Dense(units, activation=act,
                             kernel_initializer=initializer,
                             use_bias=True, name="Output_layer"))
    else:
        model_deep.add(Dense(units, activation=act,
                             kernel_initializer=initializer,
                             use_bias=True, name=f"Hidden_layer_{i}"))

# ================================
#  Modelo Deep-SITAR
# ================================
class MyDeepSITAR(tf.keras.Model):
    def __init__(self, x_ref, spline_layer, deep_model, name=None):
        super().__init__(name=name)
        self.loc_net_S = spline_layer
        self.loc_net_deep = deep_model
        self.X = x_ref  # referencia (shape [1, 20] o [batch, 20])

    def call(self, x):
        rand_effects = self.loc_net_deep(x)  # [batch, 3] => (a,b,c)
        dim_x = tf.shape(self.X)[1]
        a_i = tf.tile(rand_effects[:, 0:1], (1, dim_x))
        b_i = tf.tile(rand_effects[:, 1:2], (1, dim_x))
        c_i = tf.tile(rand_effects[:, 2:3], (1, dim_x))
        out_spl = self.loc_net_S((self.X - b_i) * tf.math.exp(c_i))
        deep_sitar = a_i + out_spl
        return deep_sitar

    def MyELBO(self, y_true, y_pred):
        re = self.loc_net_deep(y_true)
        re_center = re - tf.reduce_mean(re, axis=0, keepdims=True)

        cov = tfp.stats.covariance(re_center)
        dim = tf.shape(cov)[0]
        cov = cov + 1e-6 * tf.eye(dim)
        precision = tf.linalg.pinv(cov)

        mat1 = tf.linalg.matmul(re_center, precision)
        mat2 = tf.linalg.matmul(mat1, re_center, transpose_b=True)
        penal_diag = tf.linalg.diag_part(mat2)
        penal = tf.reshape(penal_diag, (-1, 1))

        recon = tf.reduce_mean((y_pred - y_true) ** 2)
        reg = tf.reduce_mean(penal)
        return recon + reg

    def ret_fix(self, x):
        m = tf.reduce_mean(self.loc_net_deep(x), axis=0)
        return m.numpy()

    def ret_sig(self, x):
        s = tfp.stats.covariance(self.loc_net_deep(x))
        return s.numpy()

# ================================
#  Entrenamiento principal
# ================================
bspline_layer.trainable = True   # <- congela el spline
DeepSITAR = MyDeepSITAR(x_ref=X, spline_layer=bspline_layer, deep_model=model_deep)
his_loss_train, his_loss_val = [], []

for i in range(len(lr)):
    opt = Adam(learning_rate=lr[i], epsilon=1e-7)
    DeepSITAR.compile(optimizer=opt, loss=DeepSITAR.MyELBO)
    histo = DeepSITAR.fit(
        Y_training, Y_training,
        epochs=epp[i],
        verbose=1,
        batch_size=b_size,
        validation_data=(Y_val, Y_val)
    )
    his_loss_train.extend(histo.history['loss'])
    his_loss_val.extend(histo.history['val_loss'])

# Curvas de pérdida
plot_loss(his_loss_train, his_loss_val, chain=0,    lr_list=lr, n_id=data_s, obs=20, num_seg=num_seg)
plot_loss(his_loss_train, his_loss_val, chain=1000, lr_list=lr, n_id=data_s, obs=20, num_seg=num_seg)

time_train = time() - start_time
print('Time of Training (min):', time_train / 60.0)

# ================================
#  Evaluaciones y visualización
# ================================
# Comparación en validación
plot_comparison(
    X_val, f_val, DeepSITAR(Y_val),
    title="f (val) vs estimación",
    save_path=PLOTS_DIR / f'plot_compare_f_vs_est_{num_seg}.pdf'
)

# Dispersión f vs y_estim con línea x=y
fig, ax = plt.subplots()
yhat_val = DeepSITAR(Y_val)
ax.scatter(f_val.numpy().ravel(), yhat_val.numpy().ravel(),
           marker='o', alpha=1.0, label='f vs y_estim')
min_val_xy = float(min(np.min(f_val.numpy()), np.min(yhat_val.numpy())))
max_val_xy = float(max(np.max(f_val.numpy()), np.max(yhat_val.numpy())))
ax.plot([min_val_xy, max_val_xy], [min_val_xy, max_val_xy],
        color='red', linestyle='--', label='x=f')
ax.set_xlabel("f"); ax.set_ylabel("Y_estim")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=2)
save_and_show(fig, PLOTS_DIR / f'plot_compare_estim_{num_seg}.pdf', dpi=300)

# Visualización FINAL: bases + curva B-spline ENTRENADA
x_values = np.linspace(domain[0], domain[1], 400).astype(np.float32)
x_tensor = tf.constant(x_values, dtype=tf.float32)
fig, ax = plt.subplots(figsize=(10, 6))
basis_count = bspline_layer.num_bases
for i in range(basis_count):
    basis_y = bspline_layer.bspline_basis_tensor(x_tensor, i, degree).numpy()
    ax.plot(x_values, basis_y, linestyle='--', label=f'Basis {i+1}')
y_trained = bspline_layer(x_tensor).numpy()
ax.plot(x_values, y_trained, linewidth=2, label='Trained B-spline Curve')
ax.set_title('Trained B-spline Curve and Basis Functions')
ax.set_xlabel('x'); ax.set_ylabel('Output'); ax.legend(); ax.grid(True)
save_and_show(fig, PLOTS_DIR / f'trained_bspline_curve_{num_seg}.pdf')


# Otra comparación puntual
dat = 50  # o 170
plot_comparison(
    X, f_val[dat:dat+1, :], DeepSITAR(Y_val[dat:dat+1, :]),
    title=f"Comparación puntual (dat={dat})",
    save_path=PLOTS_DIR / f'plot_point_compare_{num_seg}_dat{dat}.pdf'
)

# Efectos fijos/covarianza
fixed_est_values = DeepSITAR.ret_fix(Y_training)
sigma_est_values = DeepSITAR.ret_sig(Y_training)
print('Fixed Values:', fixed_est_values)
print('Sigma Values:', sigma_est_values)

# Matriz cov real (dummy del ejemplo original)
cov_real = [35, 0.4587, 0.2273,
            0.4587, 1.2, -0.098432,
            0.2273, -0.098432, 0.076]
cov_real_mat = np.array(cov_real).reshape(3, 3)

# ================================
#  Exports a CSV
# ================================
X_train_np = X_training.numpy().ravel()
Y_train_np = Y_training.numpy().ravel()
f_train_np = f_training.numpy().ravel()
y_estim_np = DeepSITAR(Y_training).numpy().ravel()

X_val_np = X_val.numpy().ravel()
Y_val_np = Y_val.numpy().ravel()
f_val_np = f_val.numpy().ravel()
y_val_pred_np = DeepSITAR(Y_val).numpy().ravel()

cov_est = sigma_est_values.reshape(9, 1).tolist()
cov_real1 = np.array(cov_real).reshape(9, 1).tolist()

fixed_est_list = fixed_est_values.reshape(3, 1).tolist()
fixed_est_list.append([time_train / 60.0])  # adjuntamos tiempo (min) como cuarta fila

df1_val = pd.DataFrame({
    'x': X_val_np + X_mean,
    'y': Y_val_np + Y_mean,
    'f': f_val_np + f_mean,
    'y_pred': y_val_pred_np + Y_mean,
    'num_seg': num_seg,
    'nind': data_s,
})

df1_train = pd.DataFrame({
    'x': X_train_np + X_mean,
    'y': Y_train_np + Y_mean,
    'f': f_train_np + f_mean,
    'y_estim': y_estim_np + Y_mean,
    'num_seg': num_seg,
    'nind': data_s,
})

df_loss = pd.DataFrame({
    'loss_train': his_loss_train,
    'loss_val': his_loss_val,
    'num_seg': num_seg,
    'nind': data_s,
})

fixed_est_df = pd.DataFrame({'fixed_est': fixed_est_list})

cov_real_2 = pd.DataFrame({
    'cov_real': cov_real1,
    'cov_est': cov_est,
    'nind': data_s,
})

rand_eff_training = model_deep(Y_training).numpy()
rand_eff_training = pd.DataFrame(rand_eff_training, columns=["a", "b", "c"])

rand_eff_val = model_deep(Y_val).numpy()
rand_eff_val = pd.DataFrame(rand_eff_val, columns=["a", "b", "c"])

df1_val.to_csv(RESULTS_DIR / f'estim_values_val_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
df1_train.to_csv(RESULTS_DIR / f'estim_values_train_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
df_loss.to_csv(RESULTS_DIR / f'Loss_val_vs_train_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
cov_real_2.to_csv(RESULTS_DIR / f'covariance_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
fixed_est_df.to_csv(RESULTS_DIR / f'fixed_effects_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
rand_eff_training.to_csv(RESULTS_DIR / f're_training_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
rand_eff_val.to_csv(RESULTS_DIR / f're_val_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)


# ================================
# Verificación numérica: condición natural f'' ≈ 0 en extremos
# ================================
# Creamos una malla fina de evaluación
layer = BSplineLayer(num_seg=num_seg, degree=degree, domain=domain, natural=True)

x_grid = np.linspace(domain[0], domain[1], 500).astype(np.float32)

# Evaluamos la salida del spline (con pesos iniciales aleatorios)
y_values = layer(x_grid).numpy().flatten()

# Paso de malla
dx = x_grid[1] - x_grid[0]

# Derivadas numéricas
f1 = np.gradient(y_values, dx)
f2 = np.gradient(f1, dx)

print("Segunda derivada en extremo izquierdo (a):", f2[0])
print("Segunda derivada en extremo derecho (b):", f2[-1])

# Chequeo tolerancia
tol = 1e-2
print("¿Cumple condición natural en a?", abs(f2[0]) < tol)
print("¿Cumple condición natural en b?", abs(f2[-1]) < tol)


# Usamos la misma malla que para las derivadas
x_plot = np.linspace(domain[0], domain[1], 500)

plt.figure(figsize=(10, 5))
plt.plot(x_plot, f2, label="Segunda derivada")
plt.axhline(0, color="red", linestyle="--", label="Cero")
plt.scatter([domain[0], domain[1]], [f2[0], f2[-1]], color="black", zorder=5, label="Extremos")
plt.title("Verificación condición natural: f''(a)=f''(b)=0")
plt.xlabel("x")
plt.ylabel("Segunda derivada")
plt.legend()
plt.grid(True)
plt.show()
