import numpy as np
import pandas as pd


def _simpson(f, a, b, fa=None, fm=None, fb=None):
    m = (a + b) / 2.0
    if fa is None:
        fa = f(a)
    if fm is None:
        fm = f(m)
    if fb is None:
        fb = f(b)
    S = (b - a) * (fa + 4.0 * fm + fb) / 6.0
    return S, m, fa, fm, fb


def simpson_adaptativo(f, a, b, tol, max_depth=20):
    """
    Simpson adaptativo (recursivo) con control de error:
    |S_2 - S_1|/15 <= tol
    Devuelve:
      - df: tabla con subintervalos aceptados
      - info: dict con I, subintervalos, profundidad_max
    """
    if tol <= 0:
        return pd.DataFrame(), {"ok": False, "msg": "La tolerancia debe ser positiva."}
    if max_depth <= 0:
        return pd.DataFrame(), {"ok": False, "msg": "max_depth debe ser un entero positivo."}
    if a == b:
        return pd.DataFrame(), {"ok": True, "I": 0.0, "subintervalos": 0, "profundidad_max": 0}

    # Para evitar problemas si el usuario invierte límites
    if a > b:
        a, b = b, a
        sign = -1.0
    else:
        sign = 1.0

    # Cache inicial
    S, m, fa, fm, fb = _simpson(f, a, b)

    accepted_rows = []
    depth_max_used = 0

    def recurse(a, b, tol_local, S_ab, fa, fm, fb, depth):
        nonlocal accepted_rows, depth_max_used
        depth_max_used = max(depth_max_used, depth)

        m = (a + b) / 2.0
        lm = (a + m) / 2.0
        rm = (m + b) / 2.0

        flm = f(lm)
        frm = f(rm)

        S_left = (m - a) * (fa + 4.0 * flm + fm) / 6.0
        S_right = (b - m) * (fm + 4.0 * frm + fb) / 6.0
        S2 = S_left + S_right

        err_est = abs(S2 - S_ab) / 15.0

        # Aceptar o subdividir
        if (err_est <= tol_local) or (depth >= max_depth):
            accepted_rows.append(
                {
                    "a": a,
                    "b": b,
                    "m": m,
                    "S_ab": S_ab,
                    "S_left": S_left,
                    "S_right": S_right,
                    "S2": S2,
                    "err_est": err_est,
                    "depth": depth,
                    "accepted": True if err_est <= tol_local else False,
                }
            )
            # Corrección Richardson: S2 + (S2 - S1)/15
            return S2 + (S2 - S_ab) / 15.0

        # Si no se acepta, subdividir con tol/2 por lado
        left_val = recurse(a, m, tol_local / 2.0, S_left, fa, flm, fm, depth + 1)
        right_val = recurse(m, b, tol_local / 2.0, S_right, fm, frm, fb, depth + 1)
        return left_val + right_val

    I = recurse(a, b, tol, S, fa, fm, fb, depth=1)
    df = pd.DataFrame(accepted_rows)

    return df, {
        "ok": True,
        "I": float(sign * I),
        "subintervalos": int(len(df)),
        "profundidad_max": int(depth_max_used),
        "tol": float(tol),
        "max_depth": int(max_depth),
    }
