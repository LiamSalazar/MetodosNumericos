import numpy as np
import pandas as pd


def trapecio_compuesto(f, a, b, n):
    if n <= 0:
        return pd.DataFrame(), {"ok": False, "msg": "n debe ser un entero positivo."}

    x = np.linspace(a, b, n + 1)
    fx = f(x)
    h = (b - a) / n

    w = np.ones(n + 1)
    w[0] = 0.5
    w[-1] = 0.5

    contrib = h * w * fx
    I = np.sum(contrib)

    df = pd.DataFrame(
        {
            "i": np.arange(n + 1),
            "x_i": x,
            "f(x_i)": fx,
            "w_i": w,
            "h*w_i*f(x_i)": contrib,
        }
    )

    return df, {"ok": True, "I": float(I), "h": float(h), "n": int(n), "x": x, "fx": fx}



def simpson_13_compuesto(f, a, b, n):
    if n <= 0:
        return pd.DataFrame(), {"ok": False, "msg": "n debe ser un entero positivo."}
    if n % 2 != 0:
        return pd.DataFrame(), {"ok": False, "msg": "Para Simpson 1/3 compuesto, n debe ser par."}

    x = np.linspace(a, b, n + 1)
    fx = f(x)
    h = (b - a) / n

    w = np.zeros(n + 1)
    w[0] = 1
    w[-1] = 1
    w[1:-1:2] = 4
    w[2:-1:2] = 2

    contrib = (h / 3.0) * w * fx
    I = np.sum(contrib)

    df = pd.DataFrame(
        {
            "i": np.arange(n + 1),
            "x_i": x,
            "f(x_i)": fx,
            "w_i": w,
            "(h/3)*w_i*f(x_i)": contrib,
        }
    )

    return df, {"ok": True, "I": float(I), "h": float(h), "n": int(n), "x": x, "fx": fx}



def simpson_38_compuesto(f, a, b, n):
    if n <= 0:
        return pd.DataFrame(), {"ok": False, "msg": "n debe ser un entero positivo."}
    if n % 3 != 0:
        return pd.DataFrame(), {"ok": False, "msg": "Para Simpson 3/8 compuesto, n debe ser múltiplo de 3."}

    x = np.linspace(a, b, n + 1)
    fx = f(x)
    h = (b - a) / n

    w = np.zeros(n + 1)
    w[0] = 1
    w[-1] = 1

    for i in range(1, n):
        if i % 3 == 0:
            w[i] = 2
        else:
            w[i] = 3

    contrib = (3.0 * h / 8.0) * w * fx
    I = np.sum(contrib)

    df = pd.DataFrame(
        {
            "i": np.arange(n + 1),
            "x_i": x,
            "f(x_i)": fx,
            "w_i": w,
            "(3h/8)*w_i*f(x_i)": contrib,
        }
    )

    return df, {"ok": True, "I": float(I), "h": float(h), "n": int(n), "x": x, "fx": fx}

