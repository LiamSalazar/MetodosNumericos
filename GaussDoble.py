import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

def gauss_doble(f, ax, bx, ay, by, mx, my, n):
    """
    Gauss-Legendre doble en forma compuesta:
    - [ax,bx] se parte en mx subintervalos
    - [ay,by] se parte en my subintervalos
    - n = puntos de Gauss por dimensión (2..6 recomendado)
    """
    if mx <= 0 or my <= 0:
        return pd.DataFrame(), {"ok": False, "msg": "mx y my deben ser enteros positivos."}
    if n < 2 or n > 6:
        return pd.DataFrame(), {"ok": False, "msg": "Orden n recomendado entre 2 y 6."}
    if bx <= ax or by <= ay:
        return pd.DataFrame(), {"ok": False, "msg": "Se requiere bx>ax y by>ay."}

    t, w = leggauss(n)  # nodos/pesos en [-1,1]

    hx = (bx - ax) / mx
    hy = (by - ay) / my

    I = 0.0
    rows = []

    sub_id = 0
    for ix in range(mx):
        a1 = ax + ix * hx
        b1 = a1 + hx
        cx = 0.5 * (a1 + b1)
        rx = 0.5 * (b1 - a1)

        for iy in range(my):
            a2 = ay + iy * hy
            b2 = a2 + hy
            cy = 0.5 * (a2 + b2)
            ry = 0.5 * (b2 - a2)

            sub_id += 1
            for i in range(n):
                xg = cx + rx * t[i]
                for j in range(n):
                    yg = cy + ry * t[j]
                    fij = f(xg, yg)
                    contrib = (rx * ry) * w[i] * w[j] * fij
                    I += contrib
                    rows.append([sub_id, a1, b1, a2, b2, i+1, j+1,
                                 float(t[i]), float(t[j]),
                                 float(w[i]), float(w[j]),
                                 float(xg), float(yg),
                                 float(fij), float(contrib)])

    df = pd.DataFrame(rows, columns=[
        "sub", "ax_i", "bx_i", "ay_j", "by_j",
        "i", "j", "t_i", "t_j", "w_i", "w_j",
        "x_ij", "y_ij", "f(x_ij,y_ij)", "contrib"
    ])
    info = {"ok": True, "I": float(I), "mx": mx, "my": my, "orden": n, "hx": float(hx), "hy": float(hy)}
    return df, info
