import numpy as np
import pandas as pd

def pivoteo_parcial(matriz, b):
    # Validaciones iniciales
    if matriz.shape[0] != matriz.shape[1]:
        return None, {"ok": False, "msg": "La matriz debe ser cuadrada", "solucion": None}
    
    if matriz.shape[0] != len(b):
        return None, {"ok": False, "msg": "Dimensiones no coinciden", "solucion": None}
    
    n = len(b)
    matriz_aumentada = np.column_stack([matriz.astype(float), b.astype(float)])
    iteraciones = []
    
    iteraciones.append({
        'Iteración': 0,
        'Operación': 'Matriz inicial',
        'Matriz': matriz_aumentada.copy()
    })
    
    # Eliminación hacia adelante
    matriz_escalonada, iter_count = eliminacion_recursiva(matriz_aumentada, 0, n, iteraciones, 1)
    
    # Verificar solución
    tipo_solucion, mensaje = verificar_solucion(matriz_escalonada, n)
    
    if tipo_solucion != "unica":
        df = crear_dataframe_iteraciones(iteraciones, n)
        return df, {"ok": tipo_solucion == "infinitas", "msg": mensaje, "solucion": None, "tipo": tipo_solucion}

    # --- CAMBIO AQUÍ: Inicializamos el vector de solución antes de la recursión ---
    solucion_inicial = np.zeros(n)
    solucion = sustitucion_atras_recursiva(matriz_escalonada, n - 1, n, solucion_inicial)
    
    df = crear_dataframe_iteraciones(iteraciones, n)
    return df, {"ok": True, "msg": "Sistema resuelto exitosamente", "solucion": solucion, "tipo": "unica"}

def sustitucion_atras_recursiva(matriz, i, n, solucion):
    """
    Versión corregida: 'solucion' se pasa como argumento para mantener los valores
    calculados en cada paso de la recursión.
    """
    # Caso base: Si ya procesamos la fila 0
    if i < 0:
        return solucion
    
    # Cálculo de la variable actual x[i]
    # Suma de coeficientes conocidos: sum( A[i,j] * x[j] )
    suma = sum(matriz[i, j] * solucion[j] for j in range(i + 1, n))
    
    # x[i] = (b[i] - suma) / A[i,i]
    solucion[i] = (matriz[i, n] - suma) / matriz[i, i]
    
    # Llamada recursiva para la fila de arriba (i - 1)
    return sustitucion_atras_recursiva(matriz, i - 1, n, solucion)



def eliminacion_recursiva(matriz, i, n, iteraciones, iter_count):
    if i >= n:
        iteraciones.append({
            'Iteración': iter_count,
            'Operación': 'Matriz escalonada (forma triangular superior)',
            'Matriz': matriz.copy()
        })
        return matriz, iter_count
    
    max_fila = buscar_pivote_maximo(matriz, i, n)
    if max_fila != i:
        matriz = intercambiar_filas(matriz, i, max_fila, iteraciones, iter_count)
        iter_count += 1
    
    if abs(matriz[i, i]) < 1e-10:
        return eliminacion_recursiva(matriz, i + 1, n, iteraciones, iter_count)
    
    matriz, iter_count = hacer_ceros_abajo(matriz, i, n, iteraciones, iter_count)
    return eliminacion_recursiva(matriz, i + 1, n, iteraciones, iter_count)

def buscar_pivote_maximo(matriz, col, n):
    max_fila = col
    max_valor = abs(matriz[col, col])
    for fila in range(col + 1, n):
        if abs(matriz[fila, col]) > max_valor:
            max_valor = abs(matriz[fila, col])
            max_fila = fila
    return max_fila

def intercambiar_filas(matriz, fila1, fila2, iteraciones, iter_count):
    matriz[[fila1, fila2]] = matriz[[fila2, fila1]]
    iteraciones.append({
        'Iteración': iter_count,
        'Operación': f'Intercambiar F{fila1 + 1} ↔ F{fila2 + 1}',
        'Matriz': matriz.copy()
    })
    return matriz

def hacer_ceros_abajo(matriz, i, n, iteraciones, iter_count):
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
    return matriz, iter_count

def verificar_solucion(matriz, n):
    filas_nulas = 0
    for i in range(n):
        fila_cero = all(abs(matriz[i, j]) < 1e-10 for j in range(n))
        if fila_cero:
            if abs(matriz[i, n]) > 1e-10:
                return "sin_solucion", "El sistema no tiene solución"
            else:
                filas_nulas += 1
    if filas_nulas > 0:
        return "infinitas", "El sistema tiene infinitas soluciones"
    return "unica", "El sistema tiene una solución única"

def crear_dataframe_iteraciones(iteraciones, n):
    datos = []
    for iter_data in iteraciones:
        fila = {'Iteración': iter_data['Iteración'], 'Operación': iter_data['Operación']}
        matriz = iter_data['Matriz']
        for i in range(matriz.shape[0]):
            cols_str = ' '.join([f'{val:8.4f}' for val in matriz[i]])
            fila[f'Fila {i + 1}'] = cols_str
        datos.append(fila)
    return pd.DataFrame(datos)