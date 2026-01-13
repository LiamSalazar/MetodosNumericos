# RK4_Sistemas.py
import numpy as np
import pandas as pd


def rk4_sistemas(f, x0, y0_vec, xf, h, itmax=200000):
    """
    RK4 para sistemas de EDO:
      y' = f(x, y_vec)
    donde y_vec es un vector (numpy array).

    Parámetros:
      f      : función f(x, y_vec) -> np.array (mismo tamaño que y_vec)
      x0     : inicial
      y0_vec : vector inicial (lista o np.array)
      xf     : final
      h      : paso fijo
      itmax  : máximo de pasos

    Retorna:
      df   : DataFrame con tabla (x, y1..yn)
      info : dict con resumen (xs, Ys, y_final, n_steps, ok, msg)
    """
    if h <= 0:
        return pd.DataFrame(), {"ok": False, "msg": "h debe ser > 0"}

    x0 = float(x0)
    xf = float(xf)

    if xf <= x0:
        return pd.DataFrame(), {"ok": False, "msg": "xf debe ser > x0"}

    y = np.array(y0_vec, dtype=float).reshape(-1)
    nvar = y.size

    # número de pasos para caer en xf
    n_steps = int(np.ceil((xf - x0) / h))
    if n_steps > int(itmax):
        return pd.DataFrame(), {"ok": False, "msg": "Demasiados pasos: aumente h o itmax"}

    # reajuste de h para terminar exacto en xf
    h = (xf - x0) / n_steps

    xs = np.zeros(n_steps + 1, dtype=float)
    Ys = np.zeros((n_steps + 1, nvar), dtype=float)

    xs[0] = x0
    Ys[0, :] = y

    # tabla (lista de dicts)
    rows = []
    row0 = {"k": 0, "x": x0}
    for i in range(nvar):
        row0[f"y{i+1}"] = float(y[i])
    rows.append(row0)

    for k in range(n_steps):
        x = xs[k]
        y = Ys[k, :]

        k1 = np.array(f(x, y), dtype=float).reshape(-1)
        k2 = np.array(f(x + h/2, y + (h/2)*k1), dtype=float).reshape(-1)
        k3 = np.array(f(x + h/2, y + (h/2)*k2), dtype=float).reshape(-1)
        k4 = np.array(f(x + h,   y + h*k3), dtype=float).reshape(-1)

        if k1.size != nvar or k2.size != nvar or k3.size != nvar or k4.size != nvar:
            return pd.DataFrame(rows), {"ok": False, "msg": "f(x,y) debe devolver un vector del mismo tamaño que y"}

        y_next = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x_next = x + h

        xs[k+1] = x_next
        Ys[k+1, :] = y_next

        r = {"k": k+1, "x": float(x_next)}
        for i in range(nvar):
            r[f"y{i+1}"] = float(y_next[i])
        rows.append(r)

    df = pd.DataFrame(rows)

    info = {
        "ok": True,
        "msg": "ok",
        "h": float(h),
        "n_steps": int(n_steps),
        "xs": xs,
        "Ys": Ys,
        "x_final": float(xs[-1]),
        "y_final": Ys[-1, :].copy(),
        "nvar": int(nvar),
    }

    return df, info
