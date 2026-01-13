import numpy as np
import pandas as pd

def euler(f, x0, y0, h, n):
    """
    f: función f(x, y)
    x0, y0: condiciones iniciales
    h: tamaño del paso
    n: número de iteraciones
    """
    resultados = []
    x_actual = x0
    y_actual = y0

    # Registro de la condición inicial
    resultados.append({
        "Iteración": 0,
        "x": x_actual,
        "y": y_actual,
        "f(x,y)": f(x_actual, y_actual),
        "y_siguiente": y_actual + h * f(x_actual, y_actual)
    })

    # Proceso iterativo
    for i in range(1, n + 1):
        # f(x, y) actual
        pendiente = f(x_actual, y_actual)
        
        # Fórmula de Euler: y_{n+1} = y_n + h * f(x_n, y_n)
        y_siguiente = y_actual + h * pendiente
        x_siguiente = x_actual + h
        
        # Actualizar valores
        x_actual = x_siguiente
        y_actual = y_siguiente
        
        # Guardar para la tabla (si no es la última iteración calculamos el siguiente paso esperado)
        f_val = f(x_actual, y_actual)
        y_sig_esperado = y_actual + h * f_val
        
        resultados.append({
            "Iteración": i,
            "x": round(x_actual, 4),
            "y": round(y_actual, 6),
            "f(x,y)": round(f_val, 6),
            "y_siguiente": round(y_sig_esperado, 6)
        })

    df = pd.DataFrame(resultados)
    return df, {"ok": True, "msg": "Cálculo finalizado"}