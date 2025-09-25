import numpy as np
import pandas as pd
import cmath as cm

def muller(f,p0,p1,p2, tol, itmax):
    rows = []
    error = np.nan
    p0, p1, p2 = complex(p0), complex(p1), complex(p2)
    for i in range(1, itmax+1):
        # Creación de "a" a través de la función
        a_top = (f(p0)-f(p2))*(p1-p2)-(f(p1)-f(p2))*(p0-p2) # Dividendo
        a_down = (p0-p2)*(p1-p2)*(p0-p1) # Divisor
        if a_down == 0:
            return pd.DataFrame(), {"ok": False, "msg": "División por cero en el denominador."}
        a = a_top/a_down

        # Creación de "b" a través de la función
        b_top = (f(p1)-f(p2))*(p0-p2)**2-(f(p0)-f(p2))*(p1-p2)**2
        b = b_top/a_down

        # Creación de "c"
        c = f(p2)

        # Solución de la ecuación cuadrática
        # -b ± √(b² - 4ac)
        discr = b**2 - 4*a*c # Discriminante
        sqrt_discr = cm.sqrt(discr)
        denom1 = b + sqrt_discr
        denom2 = b - sqrt_discr
        if abs(denom1) > abs(denom2):
            denom = denom1
        else:
            denom = denom2
        if denom == 0:
            return pd.DataFrame(), {"ok": False, "msg": "División por cero en el denominador."}
        p3 = p2 - (2*c)/denom # Nueva aproximación
        error = np.nan if i == 1 else abs(p3 - p2) # Error absoluto (nulo o la resta)
        rows.append(dict(i=i, p0=p0, p1=p1, p2=p2, p3=p3, f_p0=f(p0), f_p1=f(p1), f_p2=f(p2), f_p3=f(p3), a=a, b=b, c=c, error=error))
        p0, p1, p2 = p1, p2, p3 # Actualización de los puntos
        if not np.isnan(error) and error < tol:
            break
    df = pd.DataFrame(rows)
    raiz = df.iloc[-1]["p3"]
    if abs(raiz.imag) < 1e-14:  # tolerancia para limpiar parte imaginaria
        raiz = raiz.real

    return df, {
        "ok": True,
        "raiz": raiz,
        "iter": len(df),
        "error": df.iloc[-1]["error"]
    }
