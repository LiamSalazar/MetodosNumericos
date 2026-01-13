import numpy as np
import pandas as pd

def algoritmo_neville(x_vals, y_vals, x0):
    n = len(x_vals)
    P = np.zeros((n, n))
    
    for i in range(n):
        P[i][0] = y_vals[i]
        
    for j in range(1, n):
        for i in range(j, n):
            numerador = ((x0 - x_vals[i-j]) * P[i][j-1]) - ((x0 - x_vals[i]) * P[i-1][j-1])
            denominador = x_vals[i] - x_vals[i-j]
            P[i][j] = numerador / denominador
            
    columnas = [f"P_{i}" for i in range(n)]
    tabla_neville = pd.DataFrame(P, columns=columnas)
    
    resultado = P[n-1][n-1]
    
    return tabla_neville, resultado