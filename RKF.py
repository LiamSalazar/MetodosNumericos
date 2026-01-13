import numpy as np
import pandas as pd


def rkf45(f, x0, y0, xf, h0, tol, itmax, h_min=1e-12, h_max=1.0):
    """
    RKF45 (Runge–Kutta–Fehlberg 4(5)) adaptativo para y' = f(x,y).

    Parámetros:
      f      : función f(x,y)
      x0,y0  : condición inicial
      xf     : punto final
      h0     : paso inicial
      tol    : tolerancia (error absoluto por paso)
      itmax  : máximo de intentos de paso (incluye rechazos)
      h_min  : paso mínimo permitido
      h_max  : paso máximo permitido

    Retorna:
      df   : DataFrame con la tabla de pasos (aceptados y rechazados)
      info : dict con resumen y la solución (x_final, y_final, etc.)
    """

    # Coeficientes Fehlberg (RKF45 clásico)
    a2 = 1/4
    a3 = 3/8
    a4 = 12/13
    a5 = 1
    a6 = 1/2

    b21 = 1/4

    b31 = 3/32
    b32 = 9/32

    b41 = 1932/2197
    b42 = -7200/2197
    b43 = 7296/2197

    b51 = 439/216
    b52 = -8
    b53 = 3680/513
    b54 = -845/4104

    b61 = -8/27
    b62 = 2
    b63 = -3544/2565
    b64 = 1859/4104
    b65 = -11/40

    # Pesos orden 4 y 5
    c4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0], dtype=float)
    c5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55], dtype=float)

    x = float(x0)
    y = float(y0)
    xf = float(xf)

    direction = 1.0 if xf >= x else -1.0
    h = float(abs(h0)) * direction

    safety = 0.9
    eps = 1e-30

    steps = 0
    accepted = 0
    rejected = 0

    xs = [x]
    ys = [y]

    rows = []

    while steps < int(itmax):
        steps += 1

        # Ajuste para no pasarse de xf
        if (direction > 0 and x + h > xf) or (direction < 0 and x + h < xf):
            h = xf - x

        # Cálculo de k's
        k1 = f(x, y)
        k2 = f(x + a2*h, y + h*(b21*k1))
        k3 = f(x + a3*h, y + h*(b31*k1 + b32*k2))
        k4 = f(x + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(x + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(x + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))

        ks = np.array([k1, k2, k3, k4, k5, k6], dtype=float)

        y4 = y + h * float(np.dot(c4, ks))
        y5 = y + h * float(np.dot(c5, ks))

        err = abs(y5 - y4)
        ok = err <= tol

        rows.append({
            "step": steps,
            "x": x,
            "y": y,
            "h": h,
            "y4": y4,
            "y5": y5,
            "err_est": err,
            "accepted": ok
        })

        if ok:
            x = x + h
            y = y5
            xs.append(x)
            ys.append(y)
            accepted += 1

            if x == xf:
                break
        else:
            rejected += 1

        # Actualizar h
        if err == 0:
            factor = 2.0
        else:
            factor = safety * (tol / (err + eps))**(1/5)

        factor = min(5.0, max(0.2, factor))
        h = h * factor

        # Limitar h
        if abs(h) > h_max:
            h = np.sign(h) * h_max

        if abs(h) < h_min:
            df = pd.DataFrame(rows)
            return df, {
                "ok": False,
                "msg": "El paso h cayó por debajo de h_min.",
                "x_final": x,
                "y_final": y,
                "steps": steps,
                "accepted": accepted,
                "rejected": rejected,
                "xs": np.array(xs),
                "ys": np.array(ys),
            }

    converged = (x == xf)
    df = pd.DataFrame(rows)

    return df, {
        "ok": True,
        "converged": converged,
        "x_final": x,
        "y_final": y,
        "steps": steps,
        "accepted": accepted,
        "rejected": rejected,
        "xs": np.array(xs),
        "ys": np.array(ys),
        "tol": tol
    }
