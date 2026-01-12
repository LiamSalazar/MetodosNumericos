import numpy as np
import pandas as pd

def pivoteo_escalonado(matriz, b):
    # Validaciones iniciales
    if matriz.shape[0] != matriz.shape[1]:
        return None, {"ok": False, "msg": "La matriz debe ser cuadrada", "solucion": None}
    
    n = len(b)
    matriz_aumentada = np.column_stack([matriz.astype(float), b.astype(float)])
    
    # --- LÓGICA DE ESCALAMIENTO ---
    # Calculamos el factor de escala (s) para cada fila: el valor absoluto máximo de cada fila de A
    s = np.array([np.max(np.abs(matriz[i, :])) for i in range(n)])
    
    # Verificar si hay filas de ceros
    if any(s == 0):
        return None, {"ok": False, "msg": "La matriz es singular (fila de ceros)", "solucion": None}

    iteraciones = []
    iteraciones.append({
        'Iteración': 0,
        'Operación': 'Matriz inicial',
        'Matriz': matriz_aumentada.copy()
    })
    
    # Eliminación hacia adelante con pivoteo escalonado
    matriz_escalonada, iter_count = eliminacion_recursiva_escalonada(
        matriz_aumentada, 0, n, s, iteraciones, 1
    )
    
    # Verificar tipo de solución
    tipo_solucion, mensaje = verificar_solucion(matriz_escalonada, n)
    
    if tipo_solucion != "unica":
        df = crear_dataframe_iteraciones(iteraciones, n)
        return df, {"ok": False, "msg": mensaje, "solucion": None, "tipo": tipo_solucion}
    
    # Sustitución hacia atrás recursiva (usando la lógica corregida anteriormente)
    solucion = sustitucion_atras_recursiva(matriz_escalonada, n - 1, n, np.zeros(n))
    
    df = crear_dataframe_iteraciones(iteraciones, n)
    return df, {
        "ok": True, 
        "msg": "Sistema resuelto con pivoteo escalonado", 
        "solucion": solucion, 
        "tipo": "unica"
    }

def eliminacion_recursiva_escalonada(matriz, i, n, s, iteraciones, iter_count):
    if i >= n:
        iteraciones.append({
            'Iteración': iter_count,
            'Operación': 'Matriz escalonada (forma triangular superior)',
            'Matriz': matriz.copy()
        })
        return matriz, iter_count
    
    # --- SELECCIÓN DE PIVOTE ESCALONADO ---
    # Buscamos la fila 'k' tal que |matriz[k,i]| / s[k] sea máximo
    # solo entre las filas de i hasta n-1
    razones = np.abs(matriz[i:n, i]) / s[i:n]
    max_relativo_idx = np.argmax(razones) + i 
    
    # Intercambiar filas en la matriz Y en el vector de escalas s
    if max_relativo_idx != i:
        matriz[[i, max_relativo_idx]] = matriz[[max_relativo_idx, i]]
        s[i], s[max_relativo_idx] = s[max_relativo_idx], s[i] # ¡Importante intercambiar escala!
        iteraciones.append({
            'Iteración': iter_count,
            'Operación': f'Intercambio Escalonado: F{i+1} ↔ F{max_relativo_idx+1}',
            'Matriz': matriz.copy()
        })
        iter_count += 1
    
    # Hacer ceros abajo
    matriz, iter_count = hacer_ceros_abajo(matriz, i, n, iteraciones, iter_count)
    
    return eliminacion_recursiva_escalonada(matriz, i + 1, n, s, iteraciones, iter_count)

def sustitucion_atras_recursiva(matriz, i, n, solucion):
    if i < 0:
        return solucion
    suma = sum(matriz[i, j] * solucion[j] for j in range(i + 1, n))
    solucion[i] = (matriz[i, n] - suma) / matriz[i, i]
    return sustitucion_atras_recursiva(matriz, i - 1, n, solucion)

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

# (Las funciones verificar_solucion, hacer_ceros_abajo y crear_dataframe_iteraciones 
# son las mismas del código anterior)