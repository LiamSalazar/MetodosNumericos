import numpy as np
import pandas as pd

def factorizacion_lu(matriz, b):
    # Validaciones iniciales
    if matriz.shape[0] != matriz.shape[1]:
        return None, {"ok": False, "msg": "La matriz debe ser cuadrada", "solucion": None}
    
    n = matriz.shape[0]
    A = matriz.astype(float).copy()
    L = np.eye(n) # Matriz Identidad inicial para L
    U = A.copy()  # U comenzará como una copia de A
    
    iteraciones = []
    iteraciones.append({
        'Iteración': 0,
        'Operación': 'Matrices iniciales',
        'L': L.copy(),
        'U': U.copy()
    })

    # Proceso recursivo para descomposición
    try:
        L, U, iter_count = descomponer_recursivo(L, U, 0, n, iteraciones, 1)
    except ZeroDivisionError:
        return None, {"ok": False, "msg": "Error: Se encontró un pivote nulo. LU requiere pivoteo.", "solucion": None}

    # Resolución: 1. Ly = b (Sustitución hacia adelante)
    y = sustitucion_adelante(L, b)
    
    # Resolución: 2. Ux = y (Sustitución hacia atrás)
    x = sustitucion_atras_recursiva_lu(U, y, n - 1, n, np.zeros(n))

    df = crear_dataframe_lu(iteraciones, n)
    return df, {
        "ok": True, 
        "msg": "Factorización y solución completada", 
        "solucion": x, 
        "L": L, 
        "U": U
    }

def descomponer_recursivo(L, U, i, n, iteraciones, iter_count):
    if i >= n - 1:
        return L, U, iter_count

    # Verificar si el pivote es cero
    if abs(U[i, i]) < 1e-10:
        raise ZeroDivisionError("Pivote nulo detectado")

    for fila in range(i + 1, n):
        factor = U[fila, i] / U[i, i]
        L[fila, i] = factor  # Guardamos el multiplicador en L
        U[fila] = U[fila] - factor * U[i] # Eliminación en U
        
        iteraciones.append({
            'Iteración': iter_count,
            'Operación': f'Eliminación U: F{fila+1} - ({factor:.2f})*F{i+1}',
            'L': L.copy(),
            'U': U.copy()
        })
        iter_count += 1
        
    return descomponer_recursivo(L, U, i + 1, n, iteraciones, iter_count)

def sustitucion_adelante(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        suma = sum(L[i, j] * y[j] for j in range(i))
        y[i] = (b[i] - suma) / L[i, i]
    return y

def sustitucion_atras_recursiva_lu(U, y, i, n, solucion):
    if i < 0:
        return solucion
    suma = sum(U[i, j] * solucion[j] for j in range(i + 1, n))
    solucion[i] = (y[i] - suma) / U[i, i]
    return sustitucion_atras_recursiva_lu(U, y, i - 1, n, solucion)

def crear_dataframe_lu(iteraciones, n):
    datos = []
    for it in iteraciones:
        res = {'Iteración': it['Iteración'], 'Operación': it['Operación']}
        # Formatear L y U para visualización en el DF
        for r in range(n):
            res[f'L Fila {r+1}'] = ' '.join([f'{val:6.2f}' for val in it['L'][r]])
            res[f'U Fila {r+1}'] = ' '.join([f'{val:6.2f}' for val in it['U'][r]])
        datos.append(res)
    return pd.DataFrame(datos)