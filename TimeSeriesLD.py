# ================================================
#   MODELO LSTM / GRU + MEDIA POBLACIONAL + PLOTS
# ================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dense, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ================================
#   1. Cargar el CSV
# ================================

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

for d in (DATA_DIR, PLOTS_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)

data_s = 500
train_frac = 0.8
t_s = int(data_s * train_frac)

csv_path = DATA_DIR / f"simul_data_{data_s}_20new.csv"
df = pd.read_csv(csv_path)

X_raw = df["age"].values.reshape((data_s, 20))
Y_raw = df["y"].values.reshape((data_s, 20))
f_raw = df["height"].values.reshape((data_s, 20))  # curva real

# Normalización
X_mean = np.mean(X_raw)
Y_mean = np.mean(Y_raw)
f_mean = np.mean(f_raw)

X = (X_raw - X_mean).astype(np.float32)
Y = (Y_raw - Y_mean).astype(np.float32)

# Dar forma para RNN: (N, 20, 1)
X_rnn = X[:, :, None]
Y_rnn = Y[:, :, None]

X_train = X_rnn[:t_s]
Y_train = Y_rnn[:t_s]
X_val = X_rnn[t_s:]
Y_val = Y_rnn[t_s:]

# ================================
#  2. Definir la arquitectura
# ================================

use_gru = True   # cambia a False para usar LSTM

inputs = Input(shape=(20, 1))

if use_gru:
    h = GRU(64, return_sequences=True)(inputs)
    h = GRU(32, return_sequences=True)(h)
else:
    h = LSTM(64, return_sequences=True)(inputs)
    h = LSTM(32, return_sequences=True)(h)

outputs = Dense(1)(h)
model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")

model.summary()

# ================================
#  3. Entrenar
# ================================

history = model.fit(
    X_train, Y_train,
    epochs=80,
    batch_size=32,
    validation_data=(X_val, Y_val),
    verbose=1
)

# ================================
#  4. Predicción (todos los individuos)
# ================================

Y_pred = model.predict(X_rnn)     # en escala centrada
Y_pred_real = Y_pred.squeeze() + Y_mean

# ================================
#  5. Estimación de la media poblacional
# ================================

mean_pop_pred = Y_pred_real.mean(axis=0)         # (20,)
mean_pop_real = Y_raw.mean(axis=0)               # (20,)

# ================================
#  6. Plots de la media poblacional
# ================================

plt.figure(figsize=(7,4))
plt.plot(X_raw.mean(axis=0), mean_pop_real, "o-", label="Media real poblacional")
plt.plot(X_raw.mean(axis=0), mean_pop_pred, ">-", label="Media predicha poblacional")
plt.xlabel("Edad promedio")
plt.ylabel("Altura")
plt.title("Curva poblacional: Real vs Predicha")
plt.legend()
plt.grid()
plt.savefig(PLOTS_DIR / "mean_population_curve.png", dpi=300)
plt.show()

# ================================
#  7. Plots de la función de costo
# ================================

plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Curvas de costo LSTM/GRU")
plt.grid()
plt.legend()
plt.savefig(PLOTS_DIR / "loss_curves.png", dpi=300)
plt.show()

# ================================
#  8. Plot por individuo + curva poblacional
# ================================

def plot_individual_with_population(i):
    plt.figure(figsize=(7,4))
    plt.plot(X_raw[i], Y_raw[i], "o-", label="Y real individuo")
    plt.plot(X_raw[i], Y_pred_real[i], ">-", label="Y predicho individuo", alpha=0.7)
    plt.plot(X_raw[i], mean_pop_real, "--", label="Media real poblacional")
    plt.plot(X_raw[i], mean_pop_pred, "--", label="Media predicha poblacional")
    plt.xlabel("Edad")
    plt.ylabel("Altura")
    plt.title(f"Individuo {i}")
    plt.legend()
    plt.grid()
    plt.show()

# ejemplo
plot_individual_with_population(5)
plot_individual_with_population(100)

# ================================
#  9. Exportar resultados
# ================================

df_pred = pd.DataFrame({
    "x": X_raw.ravel(),
    "y_real": Y_raw.ravel(),
    "y_pred": Y_pred_real.ravel(),
    "nind": data_s
})
df_pred.to_csv(RESULTS_DIR / f"LSTM_GRU_predictions_{data_s}.csv", index=False)

pd.DataFrame({
    "mean_pop_real": mean_pop_real,
    "mean_pop_pred": mean_pop_pred
}).to_csv(RESULTS_DIR / "mean_population_estimate.csv", index=False)

print("FIN: Modelo LSTM/GRU entrenado, predicción poblacional y curvas exportadas.")
