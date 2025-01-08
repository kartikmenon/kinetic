import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .data import t as T
from .data import f_t as F_T
from .data import oos_f_t as OOS_F_T

def model_KM_V1(t, a: float):
    """
    Assumptions:
    Li, Lo at t=0 are 50%/50% in concentration
    [Lo] -> [L*] rate constant gamma = 0.02
    [Li] <-> [Lo] fwd and reverse rate constants alpha/beta are equal

    Regressors:
    Rate constant alpha
    """
    g = 0.02
    b = a
    a0 = b0 = 0.5
    Li0, Lo0 = b0/(a0+b0), a0/(a0+b0)   
    k = a + b + g
    r = 4 * a * g

    # Eigenvalues
    L1 = (-k/2) + (np.sqrt(k**2 - r))/2
    L2 = (-k/2) - (np.sqrt(k**2 - r))/2

    # Eigenvectors
    v1 = (a+L1)/b
    v2 = (a+L2)/b

    A10 = (Lo0 - v2*Li0) / (v1-v2)
    A20 = (v1*Li0 - Lo0) / (v1 - v2)
    C = 100.
    Li = np.exp(L1 * t) * A10 + np.exp(L2 * t) * A20
    Lo = v1 * np.exp(L1 * t) * A10 + v2 * np.exp(L2 * t) * A20
    return C * (Li + Lo)
    

def model_KM_V2(t, a: float, b: float):
    """
    Assumptions:
    Li, Lo at t=0 are 50%/50% in concentration
    [Lo] -> [L*] rate constant gamma = 0.02
    
    Regressors:
    Rate constants alpha, beta
    """
    g = 0.02
    a0 = b0 = 0.5
    Li0, Lo0 = b0/(a0+b0), a0/(a0+b0)   
    k = a + b + g
    r = 4 * a * g

    # Eigenvalues
    L1 = (-k/2) + (np.sqrt(k**2 - r))/2
    L2 = (-k/2) - (np.sqrt(k**2 - r))/2

    # Eigenvectors
    v1 = (a+L1)/b
    v2 = (a+L2)/b

    A10 = (Lo0 - v2*Li0) / (v1-v2)
    A20 = (v1*Li0 - Lo0) / (v1 - v2)
    C = 100.
    Li = np.exp(L1 * t) * A10 + np.exp(L2 * t) * A20
    Lo = v1 * np.exp(L1 * t) * A10 + v2 * np.exp(L2 * t) * A20
    return C * (Li + Lo)

def model_KM_V3(t, a: float, b: float, g: float):
    """
    Assumptions:
    Li, Lo at t=0 are 50%/50% in concentration
    
    Regressors:
    Rate constants alpha, beta, gamma
    """
    a0 = b0 = 0.5
    Li0, Lo0 = b0/(a0+b0), a0/(a0+b0)   
    k = a + b + g
    r = 4 * a * g

    # Eigenvalues
    L1 = (-k/2) + (np.sqrt(k**2 - r))/2
    L2 = (-k/2) - (np.sqrt(k**2 - r))/2

    # Eigenvectors
    v1 = (a+L1)/b
    v2 = (a+L2)/b

    A10 = (Lo0 - v2*Li0) / (v1-v2)
    A20 = (v1*Li0 - Lo0) / (v1 - v2)
    C = 100.
    Li = np.exp(L1 * t) * A10 + np.exp(L2 * t) * A20
    Lo = v1 * np.exp(L1 * t) * A10 + v2 * np.exp(L2 * t) * A20
    return C * (Li + Lo)


def model_MARX_2000(t, kn1: float, kn2: float, kp1: float, kp2: float):
    """
    Assumptions:
    Li, Lo at t=0 are 50%/50% in concentration
    kn2 is vanishingly small & low contribution to kinetics
    
    Regressors:
    Rate constants k(-1), k(-2), k(+1), k(+2)
    """
    g = 0.02
    a0 = b0 = 0.5
    Li0, Lo0 = b0/(a0+b0), a0/(a0+b0)   
    k = kn1 + kn2 + kp1 + kp2
    r = 4 * (kp1 * (kp2 + kn2) + kn1 * kn2) / (k**2)

    # Eigenvalues
    L1 = (k/2) * (np.sqrt(1 - r) - 1)
    L2 = -(k/2) * (np.sqrt(1 - r) + 1)

    # Eigenvectors
    f1 = (kp1 + L1) / kn1
    f2 = (kp1 + L2) / kn1
    C = 100.
    A10 = (Lo0 - f2*Li0) / (f1 - f2)
    A20 = (f1*Li0 - Lo0) / (f1 - f2)
    A1 = A10 + (C * kn2) * (1 - np.exp(-L1 * t)) / (L1 * (f1 - f2))
    A2 = A20 + (C * kn2) * (1 - np.exp(-L2 * t)) / (L1 * (f2 - f1))
    B1 = f1 * A1
    B2 = f2 * A2
    
    Li = np.exp(L1 * t) * A1 + np.exp(L2 * t) * A2
    Lo = np.exp(L1 * t) * B1 + np.exp(L2 * t) * B2
    return C * (Li + Lo)

def assess(y, x_m, dim_n=1):
    """
    Find usual suspects for goodness of fit:
    - R^2
    - Adjusted R^2 (penalises for # of terms fit)
    - MSE 
    - AIC, BIC
    """
    
    r = y - x_m
    r_sq = 1 - np.sum(r**2) / np.sum((y - np.mean(y))**2)
    mse = np.mean(r**2)
    r_sq_adj = 1 - (1 - r_sq) * (len(y) - 1) / (len(y) - dim_n - 1)
    aic = len(y) * np.log(np.sum(r**2) / len(y)) + 2 * dim_n
    bic = len(y) * np.log(np.sum(r**2) / len(y)) + dim_n * np.log(len(y))
    
    return {
        "rez": r, "r2": r_sq, "r2_adj": r_sq_adj, "mse": mse, "aic": aic, "bic": bic
    }

fit_1, fit_2, fit_3, fit_mx = [], [], [], []
fig, axs = plt.subplots(4, 2, figsize=(12, 6))
tx = np.linspace(0, 810, num=100)

for i, s_i in enumerate(F_T()):
    y_data = np.array(s_i) #/ sample[0]  # Normalize to start at 1

    p0_V1 = [0.01]
    p0_V2 = [0.01, 0.01]
    p0_V3 = [0.01, 0.01, 0.2]
    p0_MX = [0.01, 0.00001, 0.2, 0.01]

    popt_V1, pcov_V1 = curve_fit(model_KM_V1, T(), y_data, p0=p0_V1)
    popt_V2, pcov_V2 = curve_fit(model_KM_V2, T(), y_data, p0=p0_V2)
    popt_V3, pcov_V3 = curve_fit(model_KM_V3, T(), y_data, p0=p0_V3)
    fit_1.append(popt_V1)
    fit_2.append(popt_V2)
    fit_3.append(popt_V3)
    j = int(np.floor(i / (len(F_T()) / 2)))
    axs[i%4, j].plot(T(), F_T()[i], label=f"Run {i + 1}")
    axs[i%4, j].plot(tx, model_KM_V1(tx, *popt_V1), '--', label=f"Run {i+1} Fit - V1")
    axs[i%4, j].plot(tx, model_KM_V2(tx, *popt_V2), '--', label=f"Run {i+1} Fit - V2")
    axs[i%4, j].plot(tx, model_KM_V3(tx, *popt_V3), '--', label=f"Run {i+1} Fit - V3")
    axs[i%4, j].grid(True)
    axs[i%4, j].legend()
    axs[i%4, j].set_ylabel("F(t)")
    axs[i%4, j].set_xlabel("Time")

plt.title("Sample of 8 - Training")
plt.show()

v1_a = sum(a[0] for a in fit_1) / 8
v2_a = sum(a[0] for a in fit_2) / 8
v3_a = sum(a[0] for a in fit_3) / 8
v2_b = sum(b[1] for b in fit_2) / 8
v3_b = sum(b[1] for b in fit_3) / 8
v3_g = sum(g[2] for g in fit_3) / 8

print("--------------- AVG MODEL COEFFICIENTS -------------------")
print(v1_a)
print(v2_a, v2_b)
print(v3_a, v3_b, v3_g)
print("----------------------------------------------------------")

fig, axs = plt.subplots(4, 4, figsize=(12, 6))
for i, s_i in enumerate(OOS_F_T()):
    y_data = np.array(s_i)
    j = int(np.floor(i / 4))
    axs[i%4, j].plot(T(), OOS_F_T()[i], label=f"Run {i+1}")
    axs[i%4, j].plot(tx, model_KM_V1(tx, v1_a), '--', label=f"Run {i+1} Fit - V1")
    axs[i%4, j].plot(tx, model_KM_V2(tx, v2_a, v2_b), '--', label=f"Run {i+1} Fit - V2")
    axs[i%4, j].plot(tx, model_KM_V3(tx, v3_a, v3_b, v3_g), '--', label=f"Run {i+1} Fit - V3")
    axs[i%4, j].grid(True)
    axs[i%4, j].legend()
    axs[i%4, j].set_ylabel("F(t)")
    axs[i%4, j].set_xlabel("Time")

plt.title("Sample of 14 - Testing")
plt.show()

r2_1, r2_2, r2_3 = [], [], []
r2a_1, r2a_2, r2a_3 = [], [], []
for i, s_i in enumerate(OOS_F_T()):
    y_data = np.array(s_i)
    r1 = assess(y_data, model_KM_V1(T(), v1_a), dim_n=1)
    r2 = assess(y_data, model_KM_V2(T(), v2_a, v2_b), dim_n=2)
    r3 = assess(y_data, model_KM_V3(T(), v3_a, v3_b, v3_g), dim_n=3)
    r2_1.append(r1['r2'])
    r2_2.append(r2['r2'])
    r2_3.append(r3['r2'])
    r2a_1.append(r1['r2_adj'])
    r2a_2.append(r2['r2_adj'])
    r2a_3.append(r3['r2_adj'])

print(*r2_3, *r2a_3, sep='\n')
