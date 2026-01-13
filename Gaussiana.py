# Gaussiana.py
import numpy as np
import pandas as pd


def _gauss_legendre_nodes_weights(order: int):
    """
    Nodos y pesos de Gauss-Legendre en [-1, 1] para order = 2..6.
    """
    if order == 2:
        t = np.array([-0.5773502691896257, 0.5773502691896257], dtype=float)
        w = np.array([1.0, 1.0], dtype=float)
    elif order == 3:
        t = np.array([-0.7745966692414834, 0.0, 0.7745966692414834], dtype=float)
        w = np.array([0.5555555555555556, 0.8888888888888888, 0.5555555555555556], dtype=float)
    elif order == 4:
        t = np.array([-0.8611363115940526, -0.3399810435848563,
                       0.3399810435848563,  0.8611363115940526], dtype=float)
        w = np.array([0.3478548451374539, 0.6521451548625461,
                      0.6521451548625461, 0.3478548451374539], dtype=float)
    elif order == 5:
        t = np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                       0.5384693101056831,  0.9061798459386640], dtype=float)
        w = np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                      0.4786286704993665, 0.2369268850561891], dtype=float)
    elif order == 6:
        t = np.array([-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                       0.2386191860831969,  0.6612093864662645,  0.9324695142031521], dtype=float)
        w = np.array([0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                      0.4679139345726910, 0.3607615730481386, 0.1713244923791704], dtype=float)
    else:
        raise ValueError("order debe estar entre 2 y 6.")
    return t, w


def gauss_legendre(f, a: float, b: float, m: int, order: int):
    """
    Cuadratura Gauss-Legendre compuesta:
    - Divide [a,b] en m subintervalos.
    - En cada subintervalo aplica Gauss-Legendre de 'order' puntos.

    Retorna:
      df: tabla por nodo (subintervalo i, nodo j)
      info: dict con resultado
    """
    try:
        a = float(a); b = float(b)
        m = int(m); order = int(order)

        if m <= 0:
            return None, {"ok": False, "msg": "m (subintervalos) debe ser un entero positivo."}
        if not (2 <= order <= 6):
            return None, {"ok": False, "msg": "order debe estar entre 2 y 6."}
        if a == b:
            return None, {"ok": False, "msg": "a y b no pueden ser iguales."}

        sign = 1.0
        if b < a:
            a, b = b, a
            sign = -1.0

        t, w = _gauss_legendre_nodes_weights(order)

        h = (b - a) / m
        rows = []
        I = 0.0

        for i in range(m):
            ai = a + i * h
            bi = ai + h

            mid = (ai + bi) / 2.0
            half = (bi - ai) / 2.0

            # Mapeo: x = mid + half * t
            x_nodes = mid + half * t
            fx = f(x_nodes)

            # Contribución: sum( w * f(x) ) * half
            contribs = (w * fx) * half
            subtotal = float(np.sum(contribs))
            I += subtotal

            for j in range(order):
                rows.append({
                    "subintervalo": i + 1,
                    "a_i": ai,
                    "b_i": bi,
                    "j": j + 1,
                    "t_j": float(t[j]),
                    "w_j": float(w[j]),
                    "x_ij": float(x_nodes[j]),
                    "f(x_ij)": float(fx[j]),
                    "contrib": float(contribs[j]),
                })

        df = pd.DataFrame(rows)
        return df, {
            "ok": True,
            "I": float(sign * I),
            "m": int(m),
            "order": int(order),
            "h": float(h),
        }

    except Exception as e:
        return None, {"ok": False, "msg": f"Ocurrió un error en Gaussiana: {e}"}
