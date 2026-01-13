import numpy as np
import pandas as pd

def p_fijo(f, a, max_its, error_max):
    error_actual = None
    i = 1
    evaluada = 0
    rows = []
    reason = "itmax"

    while(i <= max_its):

        evaluada = f(a)
        error_actual = evaluada - a
        error_actual = abs(evaluada - a)
        rows.append(dict(i=i, Pi=a, g_Pi=evaluada, error=error_actual))
        a = evaluada
        i += 1

        if(error_actual <= error_max):
            break
    df = pd.DataFrame(rows)
    meta = {
        "ok": True,
        "raiz": df.iloc[-1]["g_Pi"] if not df.empty else None,
        "iter": len(df),
        "error": df.iloc[-1]["error"] if not df.empty else None,
        "reason": reason
    }
    return df, meta
