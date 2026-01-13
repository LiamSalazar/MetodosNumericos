import sympy as sp
import numpy as np
import pandas as pd

def derivada_adelante(f, x0, h):
    return (f(x0 + h) - f(x0)) / h


def derivada_centrada(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2*h)


def richardson_derivada(funcion_str, x0, h, metodo="adelante"):
    """
    metodo:
        "adelante"  -> orden p = 1
        "centrada"  -> orden p = 2
    """
    x = sp.symbols('x')
    expr = sp.sympify(funcion_str.replace("^", "**"))
    f = sp.lambdify(x, expr, "numpy")

    derivada_exacta = sp.diff(expr, x)
    f_real = sp.lambdify(x, derivada_exacta, "numpy")
    valor_real = float(f_real(x0))

    if metodo == "adelante":
        D_h   = derivada_adelante(f, x0, h)
        D_h2  = derivada_adelante(f, x0, h/2)
        p = 1

    elif metodo == "centrada":
        D_h   = derivada_centrada(f, x0, h)
        D_h2  = derivada_centrada(f, x0, h/2)
        p = 2

    else:
        raise ValueError("Método no válido")

    richardson = (2**p * D_h2 - D_h) / (2**p - 1)

    error = abs((valor_real - richardson) / valor_real * 100) if valor_real != 0 else 0

    datos = [{
        "x0": x0,
        "h": h,
        "Método base": metodo,
        "Aprox D(h)": round(D_h, 8),
        "Aprox D(h/2)": round(D_h2, 8),
        "Richardson": round(richardson, 8),
        "Valor real": round(valor_real, 8),
        "Error %": f"{error:.6f}%"
    }]

    return pd.DataFrame(datos)
