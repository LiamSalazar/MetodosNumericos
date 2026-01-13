import pandas as pd
import numpy as np

def newton(f, derivada, a, max_its, error_max):
    error_actual = None
    i = 1
    evaluada = 0
    rows = []
    reason = "itmax"

    while(i <= max_its):

        evaluada = a - (f(a)/derivada(a))
        error_actual = evaluada - a
        error_actual = abs(evaluada - a)
        rows.append(dict(i=i, Pi=a, P_sig=evaluada, error=error_actual))
        a = evaluada
        i += 1

        if(error_actual <= error_max):
            break
    df = pd.DataFrame(rows)
    meta = {
        "ok": True,
        "raiz": df.iloc[-1]["P_sig"] if not df.empty else None,
        "iter": len(df),
        "error": df.iloc[-1]["error"] if not df.empty else None,
        "reason": reason
    }
    return df, meta
