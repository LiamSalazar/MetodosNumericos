import numpy as np
import pandas as pd

def factorizacion_cholesky(matriz, b):
    # Validaciones iniciales
    if matriz.shape[0] != matriz.shape[1]:
        return None, {"ok": False, "msg": "La matriz debe ser cuadrada", "solucion": None}
    
    # Verificar si es simétrica
    if not np.allclose(matriz, matriz.T):
        return None, {"ok": False, "msg": "La matriz no es simétrica. Cholesky requiere simetría.", "solucion": None}

    n = matriz.shape[0]
    L = np.zeros((n, n))
    iteraciones = []

    # Proceso de Cholesky
    try:
        for i in range(n):
            for j in range(i + 1):
                suma = sum(L[i, k] * L[j, k] for k in range(j))
                
                if i == j: # Elementos de la diagonal
                    val = matriz[i, i] - suma
                    if val <= 0:
                        return None, {"ok": False, "msg": "La matriz no es definida positiva.", "solucion": None}
                    L[i, j] = np.sqrt(val)
                else: # Elementos fuera de la diagonal
                    L[i, j] = (matriz[i, j] - suma) / L[j, j]
            
            iteraciones.append({
                'Iteración': i + 1,
                'Operación': f'Cálculo de columna {i + 1} de L',
                'L': L.copy()
            })

        # Resolución: 1. Ly = b (Sustitución hacia adelante)
        y = np.zeros(n)
        for i in range(n):
            suma_y = sum(L[i, j] * y[j] for j in range(i))
            y[i] = (b[i] - suma_y) / L[i, i]

        # Resolución: 2. L^T x = y (Sustitución hacia atrás)
        LT = L.T
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            suma_x = sum(LT[i, j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - suma_x) / LT[i, i]

        df = crear_dataframe_cholesky(iteraciones, n)
        return df, {"ok": True, "msg": "Factorización de Cholesky exitosa", "solucion": x, "L": L}

    except Exception as e:
        return None, {"ok": False, "msg": f"Error matemático: {str(e)}", "solucion": None}

def crear_dataframe_cholesky(iteraciones, n):
    datos = []
    for it in iteraciones:
        res = {'Iteración': it['Iteración'], 'Operación': it['Operación']}
        for r in range(n):
            res[f'L Fila {r+1}'] = ' '.join([f'{val:8.4f}' for val in it['L'][r]])
        datos.append(res)
    return pd.DataFrame(datos)