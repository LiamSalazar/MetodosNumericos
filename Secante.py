import numpy as np
import pandas as pd

# Método de Bisección
def secante(f, a, b, tol, itmax):
    rows = []
    for i in range(1, itmax+1):
        if f(b) - f(a) == 0:
            return pd.DataFrame(rows), {"ok": False, "msg": "Denominador cero, no se puede continuar."}

        c = a - f(a) * (b - a) / (f(b) - f(a))
        error = abs(c - b) # Error absoluto (nulo o la resta)
        rows.append(dict(i=i, a=a, b=b, c=c, error=error))
        if error <= tol: # Si ya se cumple con el error se termina
            break
        a, b = b, c # Ahora a es lo que estaba en b y b es lo que estaba en c
    # DataFrame con los resultados
    df = pd.DataFrame(rows)
    return df, {"ok": True, "raiz": df.iloc[-1]["c"], "iter": len(df), "error": df.iloc[-1]["error"]}
