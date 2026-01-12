import numpy as np
import pandas as pd

def pivoteo_total(matriz, b):
    n = len(b)
    matriz_aumentada = np.column_stack([matriz.astype(float), b.astype(float)])
    
    # Registro del orden de las variables (por los intercambios de columnas)
    orden_vars = list(range(n))
    iteraciones = []
    
    iteraciones.append({
        'Iteración': 0,
        'Operación': 'Matriz inicial',
        'Matriz': matriz_aumentada.copy()
    })
    
    # Eliminación hacia adelante con pivoteo total
    matriz_escalonada, iter_count = eliminacion_recursiva_total(
        matriz_aumentada, 0, n, orden_vars, iteraciones, 1
    )
    
    # Sustitución hacia atrás
    solucion_desordenada = sustitucion_atras_recursiva(matriz_escalonada, n - 1, n, np.zeros(n))
    
    # Reordenar la solución según el intercambio de columnas
    solucion_final = np.zeros(n)
    for i, original_idx in enumerate(orden_vars):
        solucion_final[original_idx] = solucion_desordenada[i]
    
    df = crear_dataframe_iteraciones(iteraciones, n)
    return df, {
        "ok": True,
        "msg": "Sistema resuelto con pivoteo total",
        "solucion": solucion_final,
        "tipo": "unica"
    }

def eliminacion_recursiva_total(matriz, i, n, orden_vars, iteraciones, iter_count):
    if i >= n:
        iteraciones.append({
            'Iteración': iter_count,
            'Operación': 'Matriz escalonada (forma triangular superior)',
            'Matriz': matriz.copy()
        })
        return matriz, iter_count
    
    # BÚSQUEDA DEL PIVOTE TOTAL
    # Buscamos el máximo en la submatriz matriz[i:n, i:n]
    submatriz = np.abs(matriz[i:n, i:n])
    idx_max = np.unravel_index(np.argmax(submatriz), submatriz.shape)
    fila_max = idx_max[0] + i
    col_max = idx_max[1] + i
    
    # Intercambio de Filas
    if fila_max != i:
        matriz[[i, fila_max]] = matriz[[fila_max, i]]
        iteraciones.append({
            'Iteración': iter_count,
            'Operación': f'Intercambio Fila: F{i+1} ↔ F{fila_max+1}',
            'Matriz': matriz.copy()
        })
        iter_count += 1
        
    # Intercambio de Columnas (esto cambia el orden de las X)
    if col_max != i:
        matriz[:, [i, col_max]] = matriz[:, [col_max, i]]
        orden_vars[i], orden_vars[col_max] = orden_vars[col_max], orden_vars[i]
        iteraciones.append({
            'Iteración': iter_count,
            'Operación': f'Intercambio Columna: C{i+1} ↔ C{col_max+1}',
            'Matriz': matriz.copy()
        })
        iter_count += 1
    
    # Eliminación Gaussiana normal
    for fila in range(i + 1, n):
        if abs(matriz[fila, i]) > 1e-10:
            factor = matriz[fila, i] / matriz[i, i]
            matriz[fila] = matriz[fila] - factor * matriz[i]
            iteraciones.append({
                'Iteración': iter_count,
                'Operación': f'F{fila + 1} = F{fila + 1} - ({factor:.4f}) × F{i + 1}',
                'Matriz': matriz.copy()
            })
            iter_count += 1
            
    return eliminacion_recursiva_total(matriz, i + 1, n, orden_vars, iteraciones, iter_count)

def sustitucion_atras_recursiva(matriz, i, n, solucion):
    if i < 0: return solucion
    suma = sum(matriz[i, j] * solucion[j] for j in range(i + 1, n))
    solucion[i] = (matriz[i, n] - suma) / matriz[i, i]
    return sustitucion_atras_recursiva(matriz, i - 1, n, solucion)

def crear_dataframe_iteraciones(iteraciones, n):
    datos = []
    for iter_data in iteraciones:
        fila = {'Iteración': iter_data['Iteración'], 'Operación': iter_data['Operación']}
        mat = iter_data['Matriz']
        for i in range(mat.shape[0]):
            fila[f'Fila {i + 1}'] = ' '.join([f'{val:8.4f}' for val in mat[i]])
        datos.append(fila)
    return pd.DataFrame(datos)