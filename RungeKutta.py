import numpy as np
import pandas as pd

def runge_kutta(f, x0, y0, h, n, orden=4):
    resultados = []
    x, y = x0, y0
    
    resultados.append({"Paso": 0, "x": x, "y": y, "k_val": "-"})

    for i in range(1, n + 1):
        if orden == 2: # Método de Heun
            k1 = f(x, y)
            k2 = f(x + h, y + h * k1)
            y = y + (h/2) * (k1 + k2)
            ks = f"k1={k1:.4f}, k2={k2:.4f}"
            
        elif orden == 3:
            k1 = f(x, y)
            k2 = f(x + h/2, y + h/2 * k1)
            k3 = f(x + h, y - h * k1 + 2 * h * k2)
            y = y + (h/6) * (k1 + 4*k2 + k3)
            ks = f"k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}"
            
        else: # Orden 4 (El más usado)
            k1 = f(x, y)
            k2 = f(x + h/2, y + h/2 * k1)
            k3 = f(x + h/2, y + h/2 * k2)
            k4 = f(x + h, y + h * k3)
            y = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            ks = f"k1={k1:.2f}, k2={k2:.2f}, k3={k3:.2f}, k4={k4:.2f}"

        x = x + h
        resultados.append({
            "Paso": i, 
            "x": round(x, 4), 
            "y": round(y, 6), 
            "Detalles (k)": ks
        })

    return pd.DataFrame(resultados), {"ok": True}