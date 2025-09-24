import numpy as np
import pandas as pd

# Método de Bisección
def bisection(f, a, b, tol, itmax):
    if f(a)*f(b) >= 0: # Debe estar el cambio de signo dentro del intervalo
        return pd.DataFrame(), {"ok": False, "msg": "f(a)·f(b) ≥ 0. Cambia el intervalo."}
    rows = []
    fa, fb = f(a), f(b)
    c_prev = None
    for i in range(1, itmax+1):
        c = (a + b) / 2.0 # Punto medio
        fc = f(c)
        error = np.nan if c_prev is None else abs(c - c_prev) # Error absoluto (nulo o la resta)
        rows.append(dict(i=i, a=a, b=b, c=c, f_a=fa, f_b=fb, f_c=fc, error=error))
        if c_prev is not None and error <= tol: # Si ya se cumple con el error se termina
            break
        if fa * fc < 0: # Si el cambio está entre a y c
            b, fb = c, fc # Ahora b es lo que estaba en c
        else: # Si el cambio está entre c y b
            a, fa = c, fc # Ahora a es lo que estaba en c
        c_prev = c # Guardar el valor anterior de c
    # DataFrame con los resultados
    df = pd.DataFrame(rows)
    return df, {"ok": True, "raiz": df.iloc[-1]["c"], "iter": len(df), "error": df.iloc[-1]["error"]}
