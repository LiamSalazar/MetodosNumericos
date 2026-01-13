# edos_adams_bashforth.py
import numpy as np

def adams_bashforth(f, x0, y0, xf, h, order=4, max_steps=200000):
    """
    Adams–Bashforth explícito (orden 2, 3 o 4) con arranque por RK4.
    Retorna: xs, ys, info, df_steps
    df_steps es lista de dicts (para DataFrame en la app).
    """
    if h <= 0:
        raise ValueError("h debe ser > 0")
    if xf <= x0:
        raise ValueError("xf debe ser > x0")
    if order not in (2, 3, 4):
        raise ValueError("order debe ser 2, 3 o 4")

    # Ajuste de h para caer exactamente en xf
    n = int(np.ceil((xf - x0) / h))
    if n > max_steps:
        raise ValueError("Demasiados pasos: reduce xf-x0 o aumenta h.")
    h = (xf - x0) / n

    xs = np.empty(n + 1, dtype=float)
    ys = np.empty(n + 1, dtype=float)
    xs[0] = x0
    ys[0] = y0

    steps = []
    steps.append({"k": 0, "x": x0, "y": y0, "method": "init", "f": float(f(x0, y0))})

    def rk4_step(x, y, hh):
        k1 = f(x, y)
        k2 = f(x + hh/2, y + hh*k1/2)
        k3 = f(x + hh/2, y + hh*k2/2)
        k4 = f(x + hh,   y + hh*k3)
        return y + (hh/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Arranque: necesitamos valores previos según orden
    # AB2 requiere 1 punto previo, AB3 requiere 2, AB4 requiere 3
    m_start = order - 1
    for i in range(1, m_start + 1):
        xs[i] = x0 + i*h
        ys[i] = rk4_step(xs[i-1], ys[i-1], h)
        steps.append({"k": i, "x": float(xs[i]), "y": float(ys[i]), "method": "RK4-start", "f": float(f(xs[i], ys[i]))})

    # Precalcular f en puntos disponibles
    fvals = [float(f(xs[i], ys[i])) for i in range(m_start + 1)]  # f0..f_{m_start}

    # Coeficientes AB
    # y_{n+1} = y_n + h * sum(b_j f_{n-j})
    if order == 2:
        b = [3/2, -1/2]
    elif order == 3:
        b = [23/12, -16/12, 5/12]
    else:  # 4
        b = [55/24, -59/24, 37/24, -9/24]

    # Iteración AB
    for k in range(m_start, n):
        xk = xs[k]
        yk = ys[k]

        # combinación lineal de f_{k}, f_{k-1}, ...
        s = 0.0
        for j, bj in enumerate(b):
            s += bj * fvals[-1 - j]

        y_next = yk + h * s
        x_next = xk + h

        xs[k+1] = x_next
        ys[k+1] = y_next

        f_next = float(f(x_next, y_next))

        # actualizar ventana de f
        fvals.append(f_next)
        if len(fvals) > order:
            fvals.pop(0)

        steps.append({"k": k+1, "x": float(x_next), "y": float(y_next), "method": f"AB{order}", "f": float(f_next)})

    info = {
        "h": float(h),
        "n_steps": int(n),
        "order": int(order),
        "x0": float(x0),
        "xf": float(xf),
        "y0": float(y0),
        "y_final": float(ys[-1]),
        "converged": True
    }
    return xs, ys, info, steps
