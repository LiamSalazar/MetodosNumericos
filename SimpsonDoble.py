import numpy as np
import pandas as pd

def simpson_doble(f, ax, bx, ay, by, nx, ny):
    """
    Simpson 1/3 compuesto en 2D (producto tensorial).
    Requiere nx y ny pares.
    Retorna: (df, info)
    """
    if nx <= 0 or ny <= 0:
        return pd.DataFrame(), {"ok": False, "msg": "nx y ny deben ser enteros positivos."}
    if nx % 2 != 0 or ny % 2 != 0:
        return pd.DataFrame(), {"ok": False, "msg": "Simpson doble requiere nx y ny pares."}
    if bx <= ax or by <= ay:
        return pd.DataFrame(), {"ok": False, "msg": "Se requiere bx>ax y by>ay."}

    hx = (bx - ax) / nx
    hy = (by - ay) / ny

    x = ax + hx * np.arange(nx + 1)
    y = ay + hy * np.arange(ny + 1)

    # Pesos Simpson 1/3 en 1D
    wx = np.ones(nx + 1)
    wx[1:nx:2] = 4
    wx[2:nx-1:2] = 2

    wy = np.ones(ny + 1)
    wy[1:ny:2] = 4
    wy[2:ny-1:2] = 2

    # Producto tensorial
    I = 0.0
    rows = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            fij = f(x[i], y[j])
            w = wx[i] * wy[j]
            contrib = w * fij
            I += contrib
            rows.append([i, j, float(x[i]), float(y[j]), float(fij), float(w), float(contrib)])

    I *= (hx * hy) / 9.0  # (hx/3)*(hy/3) = hx*hy/9

    df = pd.DataFrame(rows, columns=["i", "j", "x_i", "y_j", "f(x_i,y_j)", "w_ij", "w_ij*f"])
    info = {"ok": True, "I": float(I), "hx": float(hx), "hy": float(hy), "nx": nx, "ny": ny}
    return df, info
