import numpy as np
import pandas as pd

# Método de Bisección
def bisection(f, a, b, tol, itmax):
    if f(a)*f(b) >= 0:
        return pd.DataFrame(), {"ok": False, "msg": "f(a)·f(b) ≥ 0. Cambia el intervalo."}
    
    rows = []
    fa, fb = f(a), f(b)
    c_prev = None
    
    for i in range(1, itmax+1):
        c = (a + b) / 2.0
        fc = f(c)
        error = np.nan if c_prev is None else abs(c - c_prev)
        
        rows.append(dict(i=i, a=a, b=b, c=c, f_a=fa, f_b=fb, f_c=fc, error=error))
        
        # 1. Verificación de raíz exacta o tolerancia
        if abs(fc) < 1e-15 or (c_prev is not None and error <= tol): 
            break
            
        # 2. Lógica de actualización de intervalos
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
            
        c_prev = c
        
    df = pd.DataFrame(rows)
    return df, {"ok": True, "raiz": c, "iter": len(df), "error": error}