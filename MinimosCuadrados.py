import numpy as np
import pandas as pd

def minimos_cuadrados(x, y, tipo="Lineal", grado=2):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    
    info = {"ok": True, "msg": "", "modelo": None, "r2": 0}
    
    if tipo == "Lineal":
        # y = a0 + a1*x
        A = np.vstack([np.ones(n), x]).T
        sol = np.linalg.lstsq(A, y, rcond=None)[0]
        a0, a1 = sol
        modelo_str = f"y = {a0:.4f} + {a1:.4f}x"
        y_pred = a0 + a1*x
        
    elif tipo == "Polinomial":
        # y = a0 + a1*x + a2*x^2 + ...
        coeffs = np.polyfit(x, y, grado)
        sol = coeffs[::-1] # Invertir para tener a0, a1, a2...
        modelo_parts = [f"{sol[i]:.4f}x^{i}" if i > 0 else f"{sol[i]:.4f}" for i in range(len(sol))]
        modelo_str = "y = " + " + ".join(modelo_parts[::-1])
        y_pred = np.polyval(coeffs, x)
        
    elif tipo == "No Lineal (Exponencial)":
        # y = a * e^(b*x)  -> ln(y) = ln(a) + b*x
        if np.any(y <= 0):
            return None, {"ok": False, "msg": "Y debe ser positivo para ajuste exponencial", "solucion": None}
        
        ln_y = np.log(y)
        A = np.vstack([np.ones(n), x]).T
        sol_ln = np.linalg.lstsq(A, ln_y, rcond=None)[0]
        a = np.exp(sol_ln[0])
        b = sol_ln[1]
        modelo_str = f"y = {a:.4f} * e^({b:.4f}x)"
        y_pred = a * np.exp(b * x)
        sol = [a, b]

    # Calcular Coeficiente de Determinación R²
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Crear DataFrame de residuos
    df_res = pd.DataFrame({
        'x': x,
        'y_real': y,
        'y_ajuste': y_pred,
        'Error^2': (y - y_pred)**2
    })
    
    info.update({"modelo": modelo_str, "r2": r2, "solucion": sol, "y_pred": y_pred})
    return df_res, info