import numpy as np
import sympy as sp
import pandas as pd

def dosTresCincoPuntos(funcion_str, x0, h, puntos, tipo):
    x = sp.symbols('x')
    expr = sp.sympify(funcion_str.replace('^', '**'))
    f = sp.lambdify(x, expr, "numpy")
    
    derivada_exacta_expr = sp.diff(expr, x)
    f_real_func = sp.lambdify(x, derivada_exacta_expr, "numpy")
    valor_real = float(f_real_func(x0))

    if puntos == "2 puntos" and tipo == "Centrada":
        raise ValueError("La derivada centrada requiere al menos 3 puntos")

    res = 0

    if puntos == "2 puntos":
        if tipo == "Adelante":
            res = (f(x0 + h) - f(x0)) / h
        if tipo == "Atrás":
            res = (f(x0) - f(x0 - h)) / h

    elif puntos == "3 puntos":
        if tipo == "Adelante":
            res = (-3*f(x0) + 4*f(x0+h) - f(x0+2*h)) / (2*h)
        if tipo == "Atrás":
            res = (3*f(x0) - 4*f(x0-h) + f(x0-2*h)) / (2*h)
        if tipo == "Centrada":
            res = (f(x0 + h) - f(x0 - h)) / (2*h)

    elif puntos == "5 puntos":
        if tipo == "Adelante":
            res = (-25*f(x0) + 48*f(x0+h) - 36*f(x0+2*h) + 16*f(x0+3*h) - 3*f(x0+4*h)) / (12*h)
        if tipo == "Atrás":
            res = (25*f(x0) - 48*f(x0-h) + 36*f(x0-2*h) - 16*f(x0-3*h) + 3*f(x0-4*h)) / (12*h)
        if tipo == "Centrada":
            res = (-f(x0+2*h) + 8*f(x0+h) - 8*f(x0-h) + f(x0-2*h)) / (12*h)

    error = abs((valor_real - res) / valor_real * 100) if valor_real != 0 else 0
    
    datos = [{
        "Punto x0": x0,
        "Aproximación": round(res, 8),
        "Valor Real": round(valor_real, 8),
        "Error %": f"{error:.6f}%"
    }]
    
    return pd.DataFrame(datos), f
