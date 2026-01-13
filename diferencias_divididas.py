def tabla_diferencias_divididas(x, y):
    import numpy as np
    import pandas as pd
    import sympy as sp

    n = len(x)
    tabla = np.full((n, n), np.nan)

    for i in range(n):
        tabla[i, 0] = y[i]

    for j in range(1, n):
        for i in range(j, n):
            tabla[i, j] = (
                tabla[i, j - 1] - tabla[i - 1, j - 1]
            ) / (x[i] - x[i - j])

    columnas = ["f(x)"] + [f"Δ^{j}f" for j in range(1, n)]
    df = pd.DataFrame(tabla, columns=columnas)
    df.insert(0, "x", x)

    df = df.where(pd.notna(df), "")

    X = sp.symbols('x')

    coeficientes = [tabla[i, i] for i in range(n)]

    P = coeficientes[0]
    producto = 1

    for i in range(1, n):
        producto *= (X - x[i - 1])
        P += coeficientes[i] * producto

    P = sp.simplify(P)

    return df, P
