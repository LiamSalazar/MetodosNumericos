# lagrange.py
import numpy as np
import pandas as pd
import sympy as sp
from sympy import symbols, lambdify, simplify

def polinomio_lagrange(x_vals, y_vals):
    x = symbols("x")
    n = len(x_vals)
    L = 0  

    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if j != i:
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        L += term

    L_simplificado = simplify(L)
    f_poly = lambdify(x, L_simplificado, "numpy")

    return L_simplificado, f_poly

def tabla_lagrange(x_vals, y_vals):
  
    x = symbols("x")
    n = len(x_vals)
    data = []

    for i in range(n):
        L_i = 1
        for j in range(n):
            if j != i:
                L_i *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        data.append({
            "i": i,
            "x_i": x_vals[i],
            "y_i": y_vals[i],
            "L_i(x)": L_i,
            "y_i*L_i(x)": y_vals[i]*L_i
        })

    df = pd.DataFrame(data)
    return df

def evaluar_lagrange(x_vals, y_vals, x0):
    _, f_poly = polinomio_lagrange(x_vals, y_vals)
    return f_poly(x0)
