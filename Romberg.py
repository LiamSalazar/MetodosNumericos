# Romberg.py
import numpy as np
import pandas as pd


def romberg(f, a: float, b: float, max_level: int, tol: float):
    """
    Romberg:
      R[k,0] = trapecio compuesto con n = 2^k subintervalos
      R[k,j] = R[k,j-1] + (R[k,j-1] - R[k-1,j-1])/(4^j - 1)

    Retorna:
      df: tabla "larga" con columnas k, j, n, R_kj, err_est
      info: dict con resultado final
    """
    try:
        a = float(a); b = float(b)
        max_level = int(max_level)
        tol = float(tol)

        if max_level < 1:
            return None, {"ok": False, "msg": "max_level debe ser >= 1."}
        if tol <= 0:
            return None, {"ok": False, "msg": "tol debe ser > 0."}
        if a == b:
            return None, {"ok": False, "msg": "a y b no pueden ser iguales."}

        sign = 1.0
        if b < a:
            a, b = b, a
            sign = -1.0

        # Matriz Romberg
        R = np.zeros((max_level + 1, max_level + 1), dtype=float)

        # Nivel 0: trapecio simple
        h = (b - a)
        R[0, 0] = 0.5 * h * (f(a) + f(b))

        rows = []
        rows.append({"k": 0, "j": 0, "n": 1, "R_kj": R[0, 0], "err_est": np.nan})

        best = R[0, 0]
        best_k = 0
        converged = False

        for k in range(1, max_level + 1):
            n = 2 ** k
            h = (b - a) / n

            # Trapecio compuesto incremental:
            # R[k,0] = 1/2 R[k-1,0] + h * sum_{i=1,3,5,...}^{n-1} f(a + i h)
            odd_indices = np.arange(1, n, 2)
            x_odd = a + odd_indices * h
            R[k, 0] = 0.5 * R[k - 1, 0] + h * np.sum(f(x_odd))

            rows.append({"k": k, "j": 0, "n": n, "R_kj": R[k, 0], "err_est": np.nan})

            # Extrapolaciones
            for j in range(1, k + 1):
                R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4 ** j - 1)

                err_est = abs(R[k, j] - R[k, j - 1])  # estimación típica (progresión en la fila)
                rows.append({"k": k, "j": j, "n": n, "R_kj": R[k, j], "err_est": err_est})

            best = R[k, k]
            best_k = k

            # Criterio de paro: diferencia diagonal (más estricto)
            diag_err = abs(R[k, k] - R[k - 1, k - 1])
            if diag_err < tol:
                converged = True
                break

        df = pd.DataFrame(rows)

        return df, {
            "ok": True,
            "I": float(sign * best),
            "k_final": int(best_k),
            "n_final": int(2 ** best_k),
            "tol": float(tol),
            "converged": bool(converged),
            "a": float(a),
            "b": float(b),
        }

    except Exception as e:
        return None, {"ok": False, "msg": f"Ocurrió un error en Romberg: {e}"}
