# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 21:20:50 2023

@author: Cyril Ferdynus
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import binom
from sklearn.linear_model import LinearRegression

#Cutoff values from Fisher et al. (2003)
sig_values = {5 : 5,
              6 : 6,
              7 : 6,
              8 : 7,
              9 : 8,
              10 : 8,
              11 : 9,
              12 : 9,
              13 : 10,
              14 : 11, 
              15 : 12,
              16 : 12,
              17 : 12,
              18 : 13,
              19 : 13,
              20 : 14,
              21 : 14,
              22 : 15,
              23 : 15}

def generate_sample(ar, ma, n_samples, scale=1.0, axis=0):     
    if np.ndim(n_samples) == 0:
        n_samples = [n_samples]
        
    new_size = tuple(n_samples)    
    e = scale * np.random.standard_normal(size=new_size)
    return signal.lfilter(ma, ar, e, axis=axis)

# Function to simulate a single multiple baseline design
# with different numbers of patients and points per patients
def SimulateOneDatasetMB(n_patients, n_points, m0 = 10.0, sd=1.0, d=0.0, a=0.0, min_baseline=5, min_phase=5):
    assert min_baseline >= 5
    assert min_phase >= 5
    assert (min_baseline >+ min_phase) <= n_points
    
    B_start = np.random.choice(np.arange(min_baseline, n_points - min_phase), n_patients)
    phases = np.array([np.hstack([np.zeros(x, dtype=int), np.ones(n_points - x, dtype=np.float32)]) for x in B_start])
    e = generate_sample([1.0, a], [1.0, 0.0], (n_patients, n_points), scale=sd)
    values =  m0 + d * phases + e
    
    return values, phases

# Simulating multiple series with alternating effects
def SimulateDatasetsMB(n_repeats, n_patients, n_points, m0=10.0, sd=1.0, effect_size=None, ar=None, min_baseline=5, min_phase=5):
    if ar is None:
        ar = np.random.normal(0.2, 0.15, n_repeats)
        ar[ar > 0.8] = 0.8
    elif not isinstance(ar, (tuple, list)):
        ar = np.array([ar]*n_repeats)
    else:
        assert len(ar) == n_repeats
        
    if effect_size is None:
        effect_size = np.random.normal(3.0, 1.0, n_repeats)
        effect_size[effect_size < 1.0] = 1.0
    elif not isinstance(ar, (tuple, list)):
        effect_size = np.array([effect_size]*n_repeats)
    else:
        assert len(effect_size) == n_repeats
    
    df_res = None
    for i in range(n_repeats):
        xy_0 = SimulateOneDatasetMB(n_patients, n_points, m0=m0, sd=sd, a=ar[i], min_baseline=min_baseline, min_phase=min_phase)
        xy_1 = SimulateOneDatasetMB(n_patients, n_points, m0=m0, sd=sd, d=effect_size[i], a=ar[i], min_baseline=min_baseline, min_phase=min_phase)
        
        df = pd.concat([pd.DataFrame(np.hstack(xy_0)), pd.DataFrame(np.hstack(xy_1))], axis=0)
        df.columns = [f"x_{i}" for i in range(1, 1 + df.shape[1] // 2)] + [f"phase_{i}" for i in range(1, 1 + df.shape[1] // 2)]
        df["simul_id"] = int(i + 1)
        df_res = pd.concat([df_res, df], axis=0)
    
    df_res["type"] = np.tile(np.hstack([np.zeros(df.shape[0] // 2), np.ones(df.shape[0] // 2)]), n_repeats).astype(int)
    df_res["patient_id"] = np.tile(np.arange(1, 1 + n_patients), n_repeats * 2).astype(int)
    df_res = df_res.set_index(["type", "simul_id", "patient_id"])
    df_res = pd.concat([df_res.iloc[:, :n_points], df_res.iloc[:, n_points:].astype(int)], axis=1)
      
    return df_res

def DualCriterion(AB, phases):
    A = AB[phases == 0]
    B = AB[phases == 1]
    
    meanline = A.mean()
    Xa = np.arange(1, 1 + len(A)).reshape((-1, 1))
    Xb = np.arange(Xa[-1] + 1, Xa[-1] + 1 + len(B)).reshape((-1, 1))
    lm = LinearRegression().fit(Xa, A)
    trendline = lm.predict(Xb)
    
    sig_points = sum((B > trendline) & (B > meanline))
    
    return sig_points, len(B)

def DualCriterionSignificant(AB, phases, alpha=0.05):
    sig_points, nb_B = DualCriterion(AB, phases)
    sig_limit = binom.ppf(1.0 - alpha, nb_B, 0.50)
    
    return (sig_points > sig_limit).astype(int)

def DualCriterionPValue(AB, phases):
    sig_points, nb_B = DualCriterion(AB, phases)
    return 1.0 - binom.cdf(sig_points, nb_B, 0.50)

def DualCriterionSigPanels(df, alpha=0.05):
    nb_points = df.shape[1] // 2
    df_sig = pd.DataFrame(df.apply(lambda x: DualCriterionSignificant(x.values[:nb_points], x.values[nb_points:], alpha), axis=1), columns=["significant"])
  
    typeI_error, power = df_sig.groupby(level=0).mean().values[:, 0]
    return typeI_error, power, df_sig
