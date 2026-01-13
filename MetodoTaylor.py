import numpy as np
import pandas as pd
import math  # Importamos el módulo math estándar

def taylor_orden_superior(funciones_derivadas, x0, y0, h, n):
    resultados = []
    x_actual = x0
    y_actual = y0
    orden = len(funciones_derivadas)

    resultados.append({
        "Iteración": 0, "x": x_actual, "y": y_actual, "Operación": "Condición Inicial"
    })

    for i in range(1, n + 1):
        termino_taylor = 0
        for k, f_der in enumerate(funciones_derivadas):
            # CAMBIO AQUÍ: Usar math.factorial en lugar de np.math
            factorial = math.factorial(k + 1)
            termino_taylor += (h**k / factorial) * f_der(x_actual, y_actual)
        
        y_siguiente = y_actual + h * termino_taylor
        x_siguiente = x_actual + h
        
        x_actual = x_siguiente
        y_actual = y_siguiente
        
        resultados.append({
            "Iteración": i,
            "x": round(x_actual, 4),
            "y": round(y_actual, 6),
            "Operación": f"Taylor Orden {orden}"
        })

    return pd.DataFrame(resultados), {"ok": True}