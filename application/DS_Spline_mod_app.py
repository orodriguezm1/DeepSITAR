# run_berkeley.py
# -*- coding: utf-8 -*-
# Fix TensorFlow Probability distutils issue
import setuptools, types, distutils
from setuptools import distutils as setuptools_distutils
distutils.version = types.SimpleNamespace(
    LooseVersion=setuptools_distutils.version.LooseVersion
)

import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
from deep_sitar_lib import Paths, TrainConfig, make_tensors_from_df, train_deep_sitar, export_all
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.initializers")

# ============================================================
#  LOAD REAL DATA (BERKELEY)
# ============================================================

df = pd.read_csv("/home/mhernandez/PycharmProjects/DeepSITAR/application/Berkeley_data.csv")
df = df.rename(columns={'height': 'y'})

n_ind = df['id'].nunique()
train_frac = 0.8

cfg = TrainConfig(
    data_s=n_ind,
    train_frac=train_frac,
    epp=[40000, 5000, 5000],
    lr=[1e-2, 1e-3, 1e-4],
    batch_size=100,
    num_seg=4,       # smoother, fewer segments for real data
    degree=3,
    lam_smooth=1e-4,
    natural=True     # use natural spline f''(ends)=0 for growth data
)

base = Path.cwd()
paths = Paths.from_base(base)

# ============================================================
#  CREATE TENSORS (AUTO-DETECT n_obs)
# ============================================================

(X_tr, Y_tr, f_tr,
 X_val, Y_val, f_val,
 X_ref, means, domain, n_obs) = make_tensors_from_df(df, cfg.data_s, cfg.train_frac)

print(f"Detected n_obs per subject = {n_obs}")

# ============================================================
#  TRAIN DEEP-SITAR
# ============================================================

model, loss_tr, loss_val, train_minutes = train_deep_sitar(
    X_tr, Y_tr, X_val, Y_val, X_ref, domain, cfg, paths
)

# ============================================================
#  EXPORT & SUMMARIZE
# ============================================================

summary = export_all(
    model, paths, cfg,
    means, X_tr, Y_tr, Y_tr, X_val, Y_val, Y_val,
    loss_tr, loss_val, train_minutes
)

print("\n=== BERKELEY DATA RESULTS ===")
print(f"Training time (min): {summary['train_minutes']:.2f}")
print("Fixed effects:", summary['fixed_est'])
print("Covariance:\n", summary['cov_est'])

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import conversion, default_converter, globalenv
from rpy2.robjects.conversion import localconverter
# --- send to R ---
with localconverter(pandas2ri.converter):
    ro.globalenv["df"] = ro.conversion.py2rpy(df)


# --- inspect in R ---
ro.r('str(df)')

# --- fit SITAR ---
ro.r('''
library(nlme)
library(dplyr)
library(glue)
library(sitar)

# Load data
print(head(df))

# Fit SITAR model correctly
fit_sitar <- try(sitar(y=y, x= age , id= id, data = df, df = num_seg), silent=TRUE)

fixed_sit=data.frame(fixed_eff=fit_sitar$coefficients$fixed,coef=names(fit_sitar$coefficients$fixed),df=num_seg,nind= data_s,model="sitar")
print(head(fixed_sit))
re_sit=data.frame(fixed_eff=fit_sitar$coefficients$random,df=num_seg,nind= data_s,model="sitar")
print(head(fixed_sit))
y_estim_sitar=data.frame(df=num_seg,nind= data_s,id=df$id, x=df$age,y_estim=fitted(fit_sitar), model="sitar")
print(head(y_estim_sitar))

cov_sitar=  data.frame(cov_est=as.numeric(VarCorr(fit_sitar)[,1]), nind=data_s, model="sitar", df=num_seg, coef=c("sigma[a]^2","sigma[b]^2","sigma[c]^2", "sigma"))
print(head(cov_sitar))
print(paste0("Fin sitar R"))
''')