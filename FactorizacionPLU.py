import numpy as np
import pandas as pd

def factorizacion_plu(matriz, b):
    # Validaciones iniciales
    if matriz.shape[0] != matriz.shape[1]:
        return None, {"ok": False, "msg": "La matriz debe ser cuadrada", "solucion": None}
    
    n = matriz.shape[0]
    A = matriz.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)
    U = A.copy()
    
    iteraciones = []
    iteraciones.append({
        'Iteración': 0,
        'Operación': 'Matrices iniciales',
        'P': P.copy(), 'L': L.copy(), 'U': U.copy()
    })

    try:
        P, L, U, iter_count = descomponer_plu_recursivo(P, L, U, 0, n, iteraciones, 1)
        
        # Resolver: Pb (ajustar el vector b según los intercambios de filas)
        b_permutado = np.dot(P, b)
        
        # 1. Ly = Pb (Sustitución hacia adelante)
        y = np.zeros(n)
        for i in range(n):
            suma = sum(L[i, k] * y[k] for k in range(i))
            y[i] = (b_permutado[i] - suma) / L[i, i]
            
        # 2. Ux = y (Sustitución hacia atrás)
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            suma = sum(U[i, k] * x[k] for k in range(i + 1, n))
            x[i] = (y[i] - suma) / U[i, i]

        df = crear_dataframe_plu(iteraciones, n)
        return df, {"ok": True, "msg": "Factorización PLU completada", "solucion": x, "P": P, "L": L, "U": U}

    except Exception as e:
        return None, {"ok": False, "msg": f"Error: {str(e)}", "solucion": None}

def descomponer_plu_recursivo(P, L, U, i, n, iteraciones, iter_count):
    if i >= n - 1:
        return P, L, U, iter_count

    # --- Pivoteo Parcial ---
    max_fila = i + np.argmax(abs(U[i:, i]))
    
    if abs(U[max_fila, i]) < 1e-12:
        raise ValueError("La matriz es singular y no se puede factorizar.")

    if max_fila != i:
        # Intercambiar en U
        U[[i, max_fila]] = U[[max_fila, i]]
        # Intercambiar en P
        P[[i, max_fila]] = P[[max_fila, i]]
        # Intercambiar filas en L (solo las columnas ya calculadas a la izquierda de i)
        if i > 0:
            L[[i, max_fila], :i] = L[[max_fila, i], :i]
            
        iteraciones.append({
            'Iteración': iter_count,
            'Operación': f'Intercambio fila {i+1} ↔ {max_fila+1}',
            'P': P.copy(), 'L': L.copy(), 'U': U.copy()
        })
        iter_count += 1

    # Eliminación de Gauss
    for fila in range(i + 1, n):
        factor = U[fila, i] / U[i, i]
        L[fila, i] = factor
        U[fila] = U[fila] - factor * U[i]
        
        iteraciones.append({
            'Iteración': iter_count,
            'Operación': f'Eliminación: F{fila+1} - ({factor:.2f})*F{i+1}',
            'P': P.copy(), 'L': L.copy(), 'U': U.copy()
        })
        iter_count += 1
        
    return descomponer_plu_recursivo(P, L, U, i + 1, n, iteraciones, iter_count)

def crear_dataframe_plu(iteraciones, n):
    datos = []
    for it in iteraciones:
        res = {'Iteración': it['Iteración'], 'Operación': it['Operación']}
        for r in range(n):
            res[f'U Fila {r+1}'] = ' '.join([f'{val:7.2f}' for val in it['U'][r]])
        datos.append(res)
    return pd.DataFrame(datos)