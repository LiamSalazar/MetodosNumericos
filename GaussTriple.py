import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

def gauss_triple(f, ax, bx, ay, by, az, bz, mx, my, mz, n):
    """
    Gauss-Legendre triple en forma compuesta:
    mx,my,mz particiones; n puntos por dimensión.
    """
    if mx <= 0 or my <= 0 or mz <= 0:
        return pd.DataFrame(), {"ok": False, "msg": "mx,my,mz deben ser enteros positivos."}
    if n < 2 or n > 6:
        return pd.DataFrame(), {"ok": False, "msg": "Orden n recomendado entre 2 y 6."}
    if bx <= ax or by <= ay or bz <= az:
        return pd.DataFrame(), {"ok": False, "msg": "Se requiere b>a en cada dimensión."}

    t, w = leggauss(n)

    hx = (bx - ax) / mx
    hy = (by - ay) / my
    hz = (bz - az) / mz

    I = 0.0
    rows = []
    sub = 0

    for ix in range(mx):
        x1 = ax + ix * hx
        x2 = x1 + hx
        cx = 0.5 * (x1 + x2)
        rx = 0.5 * (x2 - x1)

        for iy in range(my):
            y1 = ay + iy * hy
            y2 = y1 + hy
            cy = 0.5 * (y1 + y2)
            ry = 0.5 * (y2 - y1)

            for iz in range(mz):
                z1 = az + iz * hz
                z2 = z1 + hz
                cz = 0.5 * (z1 + z2)
                rz = 0.5 * (z2 - z1)

                sub += 1
                for i in range(n):
                    xg = cx + rx * t[i]
                    for j in range(n):
                        yg = cy + ry * t[j]
                        for k in range(n):
                            zg = cz + rz * t[k]
                            fval = f(xg, yg, zg)
                            contrib = (rx * ry * rz) * w[i] * w[j] * w[k] * fval
                            I += contrib
                            rows.append([sub, float(xg), float(yg), float(zg),
                                         i+1, j+1, k+1,
                                         float(fval), float(contrib)])

    df = pd.DataFrame(rows, columns=["sub", "x", "y", "z", "i", "j", "k", "f(x,y,z)", "contrib"])
    info = {"ok": True, "I": float(I), "mx": mx, "my": my, "mz": mz, "orden": n,
            "hx": float(hx), "hy": float(hy), "hz": float(hz)}
    return df, info
