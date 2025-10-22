import numpy as np
import pandas as pd

def falsa_posicion(f, a, b, tol=1e-5, itmax=100, stop="abs_x"):
    rows = []
    fa, fb = f(a), f(b)
    # Errores
    if np.isnan(fa) or np.isnan(fb):
        return pd.DataFrame(rows), {"ok": False, "msg": "f(a) o f(b) produjo NaN."}
    if fa * fb > 0:
        return pd.DataFrame(rows), {"ok": False, "msg": "El intervalo [a,b] no acota raíz (f(a)*f(b) > 0)."}
    if fb == fa:
        return pd.DataFrame(rows), {"ok": False, "msg": "Denominador cero: f(b) == f(a)."}

    c_prev = None
    reason = "itmax"
    for i in range(1, itmax + 1):
        # Fórmula: c = (a*fb - b*fa)/(fb - fa)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        # Obtener error por el criterio
        if c_prev is None:
            error = np.inf
        elif stop == "abs_f":
            error = abs(fc)
        elif stop == "rel_x":
            error = abs((c - c_prev) / c) if c != 0 else abs(c - c_prev)
        else:  
            error = abs(c - c_prev)

        rows.append(dict(i=i, a=a, b=b, c=c, f_a=fa, f_b=fb, f_c=fc, error=error))

        # paro por tolerancia
        if (stop == "abs_f" and abs(fc) <= tol) or (stop != "abs_f" and error <= tol):
            reason = "tol"
            break

        # actualización de intervalo
        prod = fa * fc
        if prod < 0:
            b, fb = c, fc
        elif prod > 0:
            a, fa = c, fc
        else:
            # raíz exacta
            reason = "root"
            break

        c_prev = c

        # evitar próxima división por cero
        if fb == fa:
            reason = "flat"
            break

    df = pd.DataFrame(rows)
    meta = {
        "ok": True,
        "raiz": df.iloc[-1]["c"] if not df.empty else None,
        "iter": len(df),
        "error": df.iloc[-1]["error"] if not df.empty else None,
        "reason": reason
    }
    return df, meta
