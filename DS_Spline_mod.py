import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
from time import time

# Establecer semillas para reproducibilidad
np.random.seed(420)
tf.random.set_seed(420)

# Inicialización
start_time = time()
data_s = 500

if data_s == 500:
    df = pd.read_csv('data/simul_data_500_20.csv')
if data_s == 1000:
    df = pd.read_csv('data/simul_data_1000_20.csv')
if data_s == 5000:
    df = pd.read_csv('data/simul_data_5000_20.csv')

t_s = int(data_s*0.8)
# Parámetros iniciales
epp = [5000,7000, 10000] # Épocas para cada fase de entrenamiento
#epp = [5000,5000, 5000] # Épocas para cada fase de entrenamiento
lr = [1e-2, 1e-3, 1e-4]     # Tasas de aprendizaje para cada fase
b_size = 100          # Tamaño del batch

# Preparación de datos
# Reshape de 'age' y 'y' a matrices de (500, 20)
X_raw = df['age'].values.reshape((data_s, 20))
Y_raw = df['y'].values.reshape((data_s, 20))
f_raw = df['height'].values.reshape((data_s, 20))

# Centramos los datos restando la media
X_mean = np.mean(X_raw)
X_training_1 = X_raw - X_mean
Y_mean = np.mean(Y_raw)
Y_training_1 = Y_raw - Y_mean
f_mean = np.mean(f_raw)
f_training_1 = f_raw - f_mean

# Convertimos a tensores de TensorFlow
X_training = tf.convert_to_tensor(X_training_1, dtype=tf.float32)[:t_s,:]
Y_training = tf.convert_to_tensor(Y_training_1, dtype=tf.float32)[:t_s,:]
f_training = tf.convert_to_tensor(f_training_1, dtype=tf.float32)[:t_s,:]

X_val = tf.convert_to_tensor(X_training_1, dtype=tf.float32)[t_s:,:]
Y_val = tf.convert_to_tensor(Y_training_1, dtype=tf.float32)[t_s:,:]
f_val = tf.convert_to_tensor(f_training_1, dtype=tf.float32)[t_s:,:]

XX = np.array(df['age']).reshape((data_s,20))[0:1,:].reshape([1,20])
X = XX - np.mean(XX)

# Definimos el dominio para la B-spline basado en X_training
domain = (np.min(X_training.numpy()), np.max(X_training.numpy()))

# Inicializador
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=.1)

# ---------------------------------------------------------------------------
###### Start Plot
# ---------------------------------------------------------------------------

def plot_comparison(X_exact, Y_exact, Y_estimation):

    plt.scatter(X_exact, Y_exact, marker='o', alpha=0.2, label='Exact')
    plt.scatter(X_exact, Y_estimation, marker='>', alpha=1., label='Estimation')
    plt.ylabel("Y")
    # Configurar la leyenda centrada en la parte inferior fuera del gráfico
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    plt.show()

def plot_loss(his_loss_train, his_loss_val, chain, lr, n_id=500, obs=20):
    """
    Función que grafica la pérdida logarítmica durante el entrenamiento.
    """
    # Convertir el historial de pérdidas a un array y seleccionar los últimos puntos basados en 'chain'
    plt.semilogy(np.asarray(his_loss_train).reshape(-1)[-chain:], label='Training Loss')
    plt.semilogy(np.asarray(his_loss_val).reshape(-1)[-chain:], label='Validation Loss')
    # Establecer el título y etiquetas del gráfico
    plt.title(f'n_id = {n_id} Obs = {obs} LR = {lr}')
    plt.ylabel("log-Loss")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    plt.savefig(f'plots/plot_estim_chain_{chain}_{num_seg}.pdf', dpi=400, bbox_inches='tight')
    # Mostrar el gráfico
    plt.show()

# ---------------------------------------------------------------------------
###### End Plot
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
###### Start Spline
# ---------------------------------------------------------------------------

# Definición de la capa B-Spline   


class BSplineLayer(tf.keras.layers.Layer):
    def __init__(self, num_seg, degree, domain, **kwargs):
        super(BSplineLayer, self).__init__(**kwargs)
        self.num_seg = num_seg
        self.degree = degree
        self.domain = domain
        self.num_bases = num_seg + degree  # Number of basis functions (control points)

        # Generate the knot sequence with extra knots outside the domain
        self.knots = np.linspace(
            domain[0] - degree * (domain[1] - domain[0]) / num_seg,
            domain[1] + degree * (domain[1] - domain[0]) / num_seg,
            self.num_bases + degree + 1  # Ensure enough knots for the degree and basis functions
        ).astype(np.float32)


        # Control points (initialized randomly)
        self.control_points = tf.Variable(
            initial_value=tf.random.normal([self.num_bases, 1]),  # Matches the number of basis functions
            trainable=True
        )

    def bspline_basis_tensor(self, x, i, k):
        """Compute B-spline basis function recursively using TensorFlow."""
        # Base case: degree 0 B-spline basis functions
        if k == 0:
            return tf.cast((self.knots[i] <= x) & (x < self.knots[i + 1]), tf.float32)
        else:
            # Ensure not to access out-of-bounds knots
            denom1 = self.knots[i + k] - self.knots[i] + 1e-8
            denom2 = self.knots[i + k + 1] - self.knots[i + 1] + 1e-8

            # Safe recursion using `tf.where` to handle bounds
            term1 = tf.where(denom1 > 0, (x - self.knots[i]) / denom1 * self.bspline_basis_tensor(x, i, k - 1), 0.0)
            term2 = tf.where(denom2 > 0, (self.knots[i + k + 1] - x) / denom2 * self.bspline_basis_tensor(x, i + 1, k - 1), 0.0)
            return term1 + term2

    def call(self, inputs):
        # Compute B-spline basis matrix
        splines = []
        for i in range(self.num_bases):  # num_bases corresponds to num_seg + degree
            spline = self.bspline_basis_tensor(inputs, i, self.degree)
            splines.append(spline)
        splines = tf.stack(splines, axis=-1)

        # Compute weighted sum of splines with control points
        weighted_splines = tf.matmul(splines, self.control_points)
        return tf.squeeze(weighted_splines, axis=-1)



# ---------------------------------------------------------------------------
###### End Spline
# ---------------------------------------------------------------------------

# Ejemplo de uso
num_seg = 5 # Número de nodos 5,8,10,15
degree = 3  # Grado del B-Spline
domain = [-4.5, 4.5]  # Dominio de la función

# Crear la capa B-Spline
bspline_layer = BSplineLayer(num_seg=num_seg, degree=degree, domain=domain)

# Llamar a la función con algunos valores de entrada

Spline_est = bspline_layer(X_training).numpy()  # Salida ajustada del B-Spline

plot_comparison(X_training,Y_training,Spline_est) ### compara con la primer estimacion del spline

# ---------------------------------------------------------------------------
###### Start deep
# ---------------------------------------------------------------------------

inp_deep = 20 # Input dimension
out_deep = 3 # Outpur dimension 3 means and covariance matriz

# # Sequential model construction
nodes_deep = [30,30]
nodes_deep.append(out_deep)
functions_deep = ['tanh','tanh']
fun_out_deep = 'linear'
functions_deep.append(fun_out_deep)
model_deep = Sequential()
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=.1) # Initial values of NN
for i in range(len(nodes_deep)):
    if i==0:
        model_deep.add(Dense(nodes_deep[i], input_shape=(inp_deep,), activation=functions_deep[i],kernel_initializer=initializer,use_bias=True,name="Input_layer"))
    elif i==(len(nodes_deep)-1):
        model_deep.add(Dense(nodes_deep[i], activation=functions_deep[i],kernel_initializer=initializer,use_bias=True,name="Ouput_layer"))
    else:
        model_deep.add(Dense(nodes_deep[i], activation=functions_deep[i],kernel_initializer=initializer,use_bias=True,name='Hidden_layer_'+str(i)))


# ---------------------------------------------------------------------------
###### End deep
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
###### Start VAE
# ---------------------------------------------------------------------------

class MyDeepSITAR(tf.keras.Model):
    def __init__(self, x_, name=None):
        super(MyDeepSITAR, self).__init__()
        self.loc_net_S = bspline_layer
        self.loc_net_deep = model_deep
        self.X = x_

    def call(self, x):
        self.rand_efects = self.loc_net_deep(x)
        #--
        dim_x = self.X.shape[1]
        a_i = tf.tile(self.rand_efects[:,0:1],(1,dim_x))
        b_i = tf.tile(self.rand_efects[:,1:2],(1,dim_x))
        c_i = tf.tile(self.rand_efects[:,2:3],(1,dim_x))
        out_spl = self.loc_net_S((self.X - b_i)*tf.math.exp(c_i))
        deep_sitar = a_i + out_spl
        return deep_sitar

# ELBO loss funtion (abs)
    # @tf.function
    def MyELBO(self,y,yp):
        rand_efect = self.loc_net_deep(y) - tf.reduce_mean(self.loc_net_deep(y),0)
        Var = tf.linalg.inv(tfp.stats.covariance(self.loc_net_deep(y)))
        # Var = tf.linalg.inv(tfp.stats.covariance(self.loc_net_deep(Y_training)))
        Mat_1 = tf.linalg.matmul(rand_efect,Var)
        Mat_2 = tf.linalg.matmul(Mat_1,tf.transpose(rand_efect))
        # penal = tf.math.log(tf.reshape(tf.linalg.diag_part(Mat_2),(-1,1)))
        penal = tf.reshape(tf.linalg.diag_part(Mat_2),(-1,1))
        log_like = tf.math.reduce_mean((yp-y)**2) + tf.math.reduce_mean(penal)
        return (log_like)

    # @tf.function
    def ret_fix(self,x):
        m = tf.reduce_mean(self.loc_net_deep(x),0)
        return m.numpy()

    def ret_sig(self,x):
        s = tfp.stats.covariance(self.loc_net_deep(x))
        return s.numpy()


# ---------------------------------------------------------------------------
###### End VAE
# ---------------------------------------------------------------------------

his_loss = []
his_loss_train = []
his_loss_val = []

DeepSITAR = MyDeepSITAR(X)
for i in range(len(lr)):
    opt = Adam(learning_rate=lr[i],epsilon=1e-16)
    DeepSITAR.compile(optimizer=opt,loss=DeepSITAR.MyELBO)
    #histo = DeepSITAR.fit(Y_training,Y_training,epochs=epp[i],verbose=1,batch_size=b_size)
    histo = DeepSITAR.fit(Y_training,Y_training, epochs=epp[i], verbose=1, batch_size=b_size, validation_data=(Y_val, Y_val))
    #his_loss.extend(histo.history['loss'])
    his_loss_train.extend(histo.history['loss'])
    his_loss_val.extend(histo.history['val_loss'])

plot_loss(his_loss_train, his_loss_val,n_id=data_s, chain=0, lr=lr)

# En el plot de la loss, chain 0 muestra toda la cadena y 1000 muestrta los últimos 1000 valores

plot_loss(his_loss_train, his_loss_val,n_id=data_s, chain=1000, lr=lr)

time_train = time() - start_time #
print('Time of Training:',time_train/60)

# ------------------------------

plot_comparison(X_val,f_val,DeepSITAR(Y_val))


#------------------------------------------


plt.scatter(f_val, DeepSITAR(Y_val), marker='o', alpha=1, label='f vs y_estim')
# Plot the line x=y
min_val = np.min(Y_val)  # Find the minimum value of Y_val
max_val = np.max(Y_val)  # Find the maximum value of Y_val
plt.plot([min_val, max_val], [min_val, max_val]  , color='red', linestyle='--', label='x=f')


plt.xlabel("f")
plt.ylabel("Y_estim")

# Configurar la leyenda centrada en la parte inferior fuera del gráfico
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
# Save the plot to a file
plt.savefig(f'plots/plot_compare_estim_{num_seg}.pdf', dpi=300, bbox_inches='tight')

plt.show()


#-------------------------------------------



bspline_layer = BSplineLayer(num_seg=num_seg, degree=degree, domain=domain)

# Generate a range of input values over the domain
x_values = np.linspace(domain[0]-4, domain[1]+4, 100)
x_tensor = tf.constant(x_values, dtype=tf.float32)

# Compute the B-spline output for the given input range
y_values = bspline_layer(x_tensor).numpy()  # Overall B-spline output for the curve

# Plot each basis function and the overall B-spline curve
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, num_seg+3))  # Color map for basis functions

# Plot individual basis functions
for i in range(num_seg+3):
    basis_y = bspline_layer.bspline_basis_tensor(x_tensor, i, degree).numpy()
    plt.plot(x_tensor, basis_y, color=colors[i], linestyle='--', label=f'Basis {i+1}')

# Plot the B-spline curve (weighted combination of basis functions)
#plt.plot(x_values, y_values, color='black', linewidth=2, label='B-spline Curve')
plt.title('B-spline Curve and Basis Functions')
plt.xlabel('x')
plt.ylabel('Output')
plt.legend()
plt.grid()
plt.show()




#-------------------------------------------


dat = 50#170
plot_comparison(X,f_val[dat:dat+1,:],DeepSITAR(Y_val[dat:dat+1,:]))


print('Fixed Values:',DeepSITAR.ret_fix(Y_training))

print('Sigma Values:',DeepSITAR.ret_sig(Y_training))

cov_real = [35  ,0.4587,  0.2273, 0.4587 , 1.2, -0.098432, 0.2273, -0.098432 , 0.076]
np.array(cov_real).reshape(3,3)



# Convert tensors to NumPy arrays

X_train_np = X_training.numpy().ravel() 
Y_train_np = Y_training.numpy().ravel() 
f_train_np = f_training.numpy().ravel() 
y_estim_np =  DeepSITAR(Y_training).numpy().ravel() 

X_val_np = X_val.numpy().ravel() 
Y_val_np = Y_val.numpy().ravel() 
f_val_np = f_val.numpy().ravel() 
y_val_pred_np =  DeepSITAR(Y_val).numpy().ravel() 


cov_est = DeepSITAR.ret_sig(Y_training).reshape(9,1).tolist()
cov_real1 = np.array(cov_real).reshape(9,1).tolist()

fix_est= DeepSITAR.ret_fix(Y_training).reshape(3,1).tolist()
fix_est.append(time_train/60)

# Create a DataFrame 

df1_val = pd.DataFrame({
    'x': X_val_np+X_mean,
    'y': Y_val_np+Y_mean,
    'f' : f_val_np+f_mean,
    'y_pred': y_val_pred_np+Y_mean,
    'num_seg': num_seg,
    'nind' : data_s,
    })

df1_train = pd.DataFrame({
    'x': X_train_np+X_mean,
    'y': Y_train_np+Y_mean,
    'f': f_train_np+f_mean,
    'y_estim': y_estim_np+Y_mean,
    'num_seg': num_seg,
    'nind' : data_s,
    })

his_loss_train_np = his_loss_train
his_loss_val_np = his_loss_val

df2= pd.DataFrame({
    'loss_train': his_loss_train_np,
    'loss_val': his_loss_val_np,
    'num_seg': num_seg,
    'nind' : data_s,
})

fix_est= pd.DataFrame({
    'fixed_est': fix_est,
    })

cov_real_2 = pd.DataFrame({
    'cov_real' : cov_real1,
    'cov_est' : cov_est,
    'nind' : data_s,
    })


rand_eff_training=model_deep(Y_training).numpy()
rand_eff_training=pd.DataFrame(rand_eff_training, columns=["a", "b", "c"])

rand_eff_val=model_deep(Y_val).numpy()
rand_eff_val=pd.DataFrame(rand_eff_val, columns=["a", "b", "c"])


# Save DataFrame as a CSV file
df1_val.to_csv(f'results/estim_values_val_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
df1_train.to_csv(f'results/estim_values_train_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
df2.to_csv(f'results/Loss_val_vs_train_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
cov_real_2.to_csv(f'results/covariance_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
fix_est.to_csv(f'results/fixed_effects_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)

rand_eff_training.to_csv(f'results/re_training_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)
rand_eff_val.to_csv(f'results/re_val_nj_20_seg_{num_seg}_nind_{data_s}_org.csv', index=False)







