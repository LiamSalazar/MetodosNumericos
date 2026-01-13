import numpy as np

def diferencias_divididas(x, y):
    n = len(x)
    coef = np.array(y, dtype=float)

    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (np.array(x[j:n]) - np.array(x[0:n-j]))

    return coef

def derivada_newton(x, y, x0):
    coef = diferencias_divididas(x, y)
    n = len(coef)

    derivada = 0.0

    for i in range(1, n):
        suma = 0.0
        for j in range(i):
            prod = 1.0
            for k in range(i):
                if k != j:
                    prod *= (x0 - x[k])
            suma += prod
        derivada += coef[i] * suma

    return derivada
