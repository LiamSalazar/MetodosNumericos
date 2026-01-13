# App.py
import streamlit as st
from sympy import symbols, sympify, lambdify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== Importa SOLO lo que ya existe en tu proyecto ====
from Bisection import bisection
from Secante import secante
from Muller import muller
from FalsaPosicion import falsa_posicion
from newton_raphson import newton
from punto_fijo import p_fijo

# ==== Importar métodos de pivoteo ====
from Pivoteparcial import pivoteo_parcial
from Pivoteesca import pivoteo_escalonado
from Pivotetotal import pivoteo_total
# ==== Importar métodos de factorizaciones ====
from FactorizacionLU import factorizacion_lu
from FactorizacionPLU import factorizacion_plu
from Cholesky import factorizacion_cholesky
# En la sección de importaciones de App.py
from MinimosCuadrados import minimos_cuadrados


# =========================
# Utilidades
# =========================
def parse_function(expr_str: str):
    """Convierte 'x**2 - 2' -> función evaluable con numpy."""
    x = symbols("x")
    expr = sympify(expr_str, dict(x=x))
    f = lambdify(x, expr, "numpy")
    return f

def plot_function(f, xmin, xmax, raiz=None):
    x = np.linspace(xmin, xmax, 400)
    y = f(x)

    fig, ax = plt.subplots()
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    ax.plot(x, y, label="f(x)")

    if raiz is not None:
        ax.plot(raiz, f(raiz), "ro", label=f"Raíz: {raiz:.4f}")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


def inject_css():
    st.markdown(
        """
        <style>
        /* Layout general */
        .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1150px; }

        /* Hero */
        .hero {
            border-radius: 18px;
            padding: 18px 18px;
            background: linear-gradient(135deg, rgba(123,97,255,0.18), rgba(0,209,255,0.10));
            border: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 14px;
        }
        .hero h1 { margin: 0; font-size: 1.55rem; }
        .hero p { margin: 6px 0 0; opacity: 0.9; }

        /* Cards */
        .card {
            border-radius: 16px;
            padding: 14px 14px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.03);
        }
        .card h3 { margin: 0 0 6px 0; font-size: 1.05rem; }
        .card small { opacity: 0.85; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(255,255,255,0.10);
        }

        /* Botones */
        .stButton > button {
            border-radius: 12px !important;
            padding: 0.6rem 1rem !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            background: rgba(123,97,255,0.20) !important;
        }
        .stButton > button:hover {
            border: 1px solid rgba(255,255,255,0.30) !important;
            transform: translateY(-1px);
        }

        /* Dataframe */
        div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def not_ready(nombre: str):
    st.markdown(
        f"""
        <div class="card">
          <h3>{nombre}</h3>
          <small>Este módulo aún no está implementado en el proyecto. Cuando lo tenga, lo conectamos aquí.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Páginas (métodos)
# =========================
def page_biseccion():
    hero("Ecuaciones de una variable · Bisección", "Encuentra una raíz en [a, b] verificando cambio de signo.")
    expr_str = st.text_input("Introduce la ecuación f(x) =", "")
    colA, colB, colC = st.columns(3)
    tol_input = colA.text_input("Tolerancia (error máximo)", "")
    itmax_input = colB.text_input("Iteraciones máximas", "")
    _ = colC.text_input("Formato", "Ej: x**3 - x - 2", disabled=True)

    col1, col2 = st.columns(2)
    a_input = col1.text_input("a (límite inferior)", "")
    b_input = col2.text_input("b (límite superior)", "")

    if st.button("Resolver", use_container_width=True):
        try:
            tol = float(tol_input)
            itmax = int(itmax_input)
            a = float(a_input)
            b = float(b_input)
            f = parse_function(expr_str)

            df, info = bisection(f, a, b, tol, itmax)
            if not info.get("ok", True):
                st.error(info["msg"])
                return

            st.subheader("Tabla de iteraciones")
            st.dataframe(df, use_container_width=True)

            st.subheader("Visualización")
            plot_function(f, a - 1, b + 1, info.get("raiz"))

            st.success(
                f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}"
            )
        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")


def page_secante():
    hero("Ecuaciones de una variable · Secante", "Aproxima la raíz usando dos aproximaciones iniciales.")
    expr_str = st.text_input("Introduce la ecuación f(x) =", "")
    colA, colB = st.columns(2)
    tol_input = colA.text_input("Tolerancia (error máximo)", "")
    itmax_input = colB.text_input("Iteraciones máximas", "")

    col1, col2 = st.columns(2)
    a_input = col1.text_input("x0", "")
    b_input = col2.text_input("x1", "")

    if st.button("Resolver", use_container_width=True):
        try:
            tol = float(tol_input)
            itmax = int(itmax_input)
            a = float(a_input)
            b = float(b_input)
            f = parse_function(expr_str)

            df, info = secante(f, a, b, tol, itmax)
            if not info.get("ok", True):
                st.error(info["msg"])
                return

            st.subheader("Tabla de iteraciones")
            st.dataframe(df, use_container_width=True)

            st.subheader("Visualización")
            plot_function(f, min(a, b) - 1, max(a, b) + 1, info.get("raiz"))

            st.success(
                f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}"
            )
        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")


def page_falsa_posicion():
    hero("Ecuaciones de una variable · Falsa Posición", "Interpolación lineal en el intervalo con cambio de signo.")
    expr_str = st.text_input("Introduce la ecuación f(x) =", "")
    colA, colB = st.columns(2)
    tol_input = colA.text_input("Tolerancia (error máximo)", "")
    itmax_input = colB.text_input("Iteraciones máximas", "")

    col1, col2 = st.columns(2)
    a_input = col1.text_input("a (límite inferior)", "")
    b_input = col2.text_input("b (límite superior)", "")

    if st.button("Resolver", use_container_width=True):
        try:
            tol = float(tol_input)
            itmax = int(itmax_input)
            a = float(a_input)
            b = float(b_input)
            f = parse_function(expr_str)

            df, info = falsa_posicion(f, a, b, tol, itmax)
            if not info.get("ok", True):
                st.error(info["msg"])
                return

            st.subheader("Tabla de iteraciones")
            st.dataframe(df, use_container_width=True)

            st.subheader("Visualización")
            plot_function(f, a - 1, b + 1, info.get("raiz"))

            st.success(
                f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}"
            )
        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")


def page_newton():
    hero("Ecuaciones de una variable · Newton", "Requiere f(x) y f'(x).")
    expr_str = st.text_input("Introduce f(x) =", "")
    derivada_str = st.text_input("Introduce f'(x) =", "")
    colA, colB = st.columns(2)
    tol_input = colA.text_input("Tolerancia (error máximo)", "")
    itmax_input = colB.text_input("Iteraciones máximas", "")
    x0_input = st.text_input("x0 (aproximación inicial)", "")

    if st.button("Resolver", use_container_width=True):
        try:
            tol = float(tol_input)
            itmax = int(itmax_input)
            x0 = float(x0_input)
            f = parse_function(expr_str)
            f_der = parse_function(derivada_str)

            df, info = newton(f, f_der, x0, itmax, tol)
            if not info.get("ok", True):
                st.error(info["msg"])
                return

            st.subheader("Tabla de iteraciones")
            st.dataframe(df, use_container_width=True)

            st.subheader("Visualización")
            plot_function(f, x0 - 2, x0 + 2, info.get("raiz"))

            st.success(
                f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}"
            )
        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")


def page_punto_fijo():
    hero("Ecuaciones de una variable · Punto Fijo", "Iteración x_{n+1} = g(x_n). (Tu módulo actual usa f(x)).")
    expr_str = st.text_input("Introduce la función (según tu implementación actual) =", "")
    colA, colB = st.columns(2)
    tol_input = colA.text_input("Tolerancia (error máximo)", "")
    itmax_input = colB.text_input("Iteraciones máximas", "")
    x0_input = st.text_input("x0 (aproximación inicial)", "")

    if st.button("Resolver", use_container_width=True):
        try:
            tol = float(tol_input)
            itmax = int(itmax_input)
            x0 = float(x0_input)
            f = parse_function(expr_str)

            df, info = p_fijo(f, x0, itmax, tol)
            if not info.get("ok", True):
                st.error(info["msg"])
                return

            st.subheader("Tabla de iteraciones")
            st.dataframe(df, use_container_width=True)

            st.subheader("Visualización")
            plot_function(f, x0 - 2, x0 + 2, info.get("raiz"))

            st.success(
                f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}"
            )
        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")


def page_muller():
    hero("Ecuaciones de una variable · Müller", "Requiere tres aproximaciones iniciales p0, p1, p2.")
    expr_str = st.text_input("Introduce la ecuación f(x) =", "")
    col1, col2, col3 = st.columns(3)
    p0str = col1.text_input("p0", "")
    p1str = col2.text_input("p1", "")
    p2str = col3.text_input("p2", "")
    colA, colB = st.columns(2)
    tol_input = colA.text_input("Tolerancia (error máximo)", "")
    itmax_input = colB.text_input("Iteraciones máximas", "")

    if st.button("Resolver", use_container_width=True):
        try:
            p0 = float(p0str)
            p1 = float(p1str)
            p2 = float(p2str)
            tol = float(tol_input)
            itmax = int(itmax_input)
            f = parse_function(expr_str)

            df, info = muller(f, p0, p1, p2, tol, itmax)
            if not info.get("ok", True):
                st.error(info["msg"])
                return

            st.subheader("Tabla de iteraciones")
            st.dataframe(df, use_container_width=True)

            st.subheader("Visualización")
            plot_function(f, min(p0, p1, p2) - 1, max(p0, p1, p2) + 1, info.get("raiz"))

            st.success(
                f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}"
            )
        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")


# =========================
# Página de Álgebra Lineal (Pivoteos)
# =========================
def page_pivoteos():
    hero("Álgebra Lineal · Sistemas de Ecuaciones Lineales", "Resuelve sistemas Ax = b usando diferentes métodos de pivoteo.")
    
    # Selector de método de pivoteo
    metodo_pivoteo = st.radio(
        "Selecciona el método de pivoteo:",
        ["Pivoteo Parcial", "Pivoteo Escalonado", "Pivoteo Total"],
        horizontal=True
    )
    
    # Descripción del método seleccionado
    descripciones = {
        "Pivoteo Parcial": "Intercambia filas buscando el máximo en la columna actual.",
        "Pivoteo Escalonado": "Usa factores de escala para cada fila antes de elegir el pivote.",
        "Pivoteo Total": "Busca el máximo en toda la submatriz (intercambia filas Y columnas)."
    }
    st.info(f"**{metodo_pivoteo}:** {descripciones[metodo_pivoteo]}")
    
    # Entrada del tamaño del sistema
    n = st.number_input("Número de ecuaciones (n)", min_value=2, max_value=10, value=3, step=1, key="n_sistema")
    n = int(n)
    
    # Matriz de coeficientes A
    st.subheader("Matriz de coeficientes A")
    matriz_inputs = []
    for i in range(n):
        cols = st.columns(n)
        fila_inputs = []
        for j in range(n):
            with cols[j]:
                val = st.number_input(f"a[{i+1},{j+1}]", value=0.0, format="%.4f", key=f"alg_a_{i}_{j}")
                fila_inputs.append(val)
        matriz_inputs.append(fila_inputs)
    
    # Vector de términos independientes b
    st.subheader("Vector de términos independientes b")
    b_inputs = []
    cols_b = st.columns(n)
    for i in range(n):
        with cols_b[i]:
            val = st.number_input(f"b[{i+1}]", value=0.0, format="%.4f", key=f"alg_b_{i}")
            b_inputs.append(val)
    
    # Botón para resolver
    if st.button("Resolver Sistema", use_container_width=True):
        try:
            A = np.array(matriz_inputs, dtype=float)
            b = np.array(b_inputs, dtype=float)
            
            # Llamar al método correspondiente
            if metodo_pivoteo == "Pivoteo Parcial":
                df, info = pivoteo_parcial(A, b)
            elif metodo_pivoteo == "Pivoteo Escalonado":
                df, info = pivoteo_escalonado(A, b)
            else:  # Pivoteo Total
                df, info = pivoteo_total(A, b)
            
            # Mostrar resultados
            if info.get("ok"):
                st.subheader("Proceso de Solución")
                st.dataframe(df, use_container_width=True)
                
                if info.get("tipo") == "unica":
                    st.success(info["msg"])
                    solucion = info.get("solucion")
                    sol_data = {"Variable": [f"x{i+1}" for i in range(n)], 
                               "Valor": [f"{solucion[i]:.6f}" for i in range(n)]}
                    st.table(sol_data)
                elif info.get("tipo") == "infinitas":
                    st.warning(info["msg"])
            else:
                st.error(info["msg"])
        except Exception as e:
            st.error(f"Error crítico: {e}")

def page_factorizacion_lu():
    hero("Álgebra Lineal · Factorización LU", "Descompone A en L (inferior) y U (superior) para resolver Ax = b.")
    
    st.info("💡 **Nota:** Este método requiere que los pivotes no sean nulos durante la eliminación.")

    # Entrada del tamaño del sistema
    n = st.number_input("Número de ecuaciones (n)", min_value=2, max_value=10, value=3, step=1, key="n_lu")
    n = int(n)
    
    # Matriz A
    st.subheader("Matriz de coeficientes A")
    matriz_inputs = []
    for i in range(n):
        cols = st.columns(n)
        fila_inputs = []
        for j in range(n):
            with cols[j]:
                val = st.number_input(f"A[{i+1},{j+1}]", value=1.0 if i==j else 0.0, format="%.2f", key=f"lu_a_{i}_{j}")
                fila_inputs.append(val)
        matriz_inputs.append(fila_inputs)
    
    # Vector b
    st.subheader("Vector b")
    b_inputs = []
    cols_b = st.columns(n)
    for i in range(n):
        with cols_b[i]:
            val = st.number_input(f"b[{i+1}]", value=0.0, format="%.2f", key=f"lu_b_{i}")
            b_inputs.append(val)
    
    if st.button("Factorizar y Resolver", use_container_width=True):
        try:
            A = np.array(matriz_inputs, dtype=float)
            b = np.array(b_inputs, dtype=float)
            
            df, info = factorizacion_lu(A, b)
            
            if info["ok"]:
                st.subheader("Proceso de Factorización (Pasos)")
                st.dataframe(df, use_container_width=True)
                
                col_l, col_u = st.columns(2)
                with col_l:
                    st.markdown("**Matriz L (Triangular Inferior)**")
                    st.write(info["L"])
                with col_u:
                    st.markdown("**Matriz U (Triangular Superior)**")
                    st.write(info["U"])
                
                st.success(info["msg"])
                st.subheader("Solución Final X")
                solucion = info["solucion"]
                sol_data = {"Variable": [f"x{i+1}" for i in range(n)], 
                           "Valor": [f"{solucion[i]:.6f}" for i in range(n)]}
                st.table(sol_data)
            else:
                st.error(info["msg"])
        except Exception as e:
            st.error(f"Error en el proceso: {e}")
def page_cholesky():
    hero("Álgebra Lineal · Factorización de Cholesky", "Factoriza A = L·Lᵀ para matrices simétricas y definidas positivas.")
    
    st.info("💡 **Requisito:** La matriz A debe ser simétrica (A = Aᵀ) y definida positiva.")

    # Entrada del tamaño del sistema
    n = st.number_input("Número de ecuaciones (n)", min_value=2, max_value=10, value=3, step=1, key="n_cholesky")
    n = int(n)
    
    # Matriz A
    st.subheader("Matriz de coeficientes A")
    matriz_inputs = []
    for i in range(n):
        cols = st.columns(n)
        fila_inputs = []
        for j in range(n):
            with cols[j]:
                # Valor por defecto 1.0 en diagonal y 0.0 fuera para evitar errores iniciales
                val = st.number_input(f"A[{i+1},{j+1}]", value=1.0 if i==j else 0.0, format="%.2f", key=f"ch_a_{i}_{j}")
                fila_inputs.append(val)
        matriz_inputs.append(fila_inputs)
    
    # --- CORRECCIÓN AQUÍ: Vector b en la interfaz normal ---
    st.subheader("Vector de términos independientes b")
    b_inputs = []
    cols_b = st.columns(n)
    for i in range(n):
        with cols_b[i]:
            val = st.number_input(f"b[{i+1}]", value=0.0, format="%.2f", key=f"ch_b_{i}")
            b_inputs.append(val)
    
    # Botón para resolver
    if st.button("Factorizar y Resolver", use_container_width=True):
        try:
            A = np.array(matriz_inputs, dtype=float)
            b = np.array(b_inputs, dtype=float)
            
            df, info = factorizacion_cholesky(A, b)
            
            if info["ok"]:
                st.subheader("Proceso de Cálculo de L")
                st.dataframe(df, use_container_width=True)
                
                col_l, col_lt = st.columns(2)
                with col_l:
                    st.markdown("**Matriz L**")
                    st.write(info["L"])
                with col_lt:
                    st.markdown("**Matriz Lᵀ (Transpuesta)**")
                    st.write(info["L"].T)
                
                st.success(info["msg"])
                st.subheader("Solución Final X")
                solucion = info["solucion"]
                sol_data = {"Variable": [f"x{i+1}" for i in range(n)], 
                           "Valor": [f"{solucion[i]:.6f}" for i in range(n)]}
                st.table(sol_data)
            else:
                st.error(info["msg"])
        except Exception as e:
            st.error(f"Error crítico: {e}")
def page_plu():
    hero("Álgebra Lineal · Factorización PLU", "Descomposición P·A = L·U con pivoteo parcial para mayor estabilidad.")
    
    n = st.number_input("Número de ecuaciones (n)", min_value=2, max_value=10, value=3, key="n_plu")
    n = int(n)
    
    # Matriz A
    st.subheader("Matriz de coeficientes A")
    matriz_inputs = []
    for i in range(n):
        cols = st.columns(n)
        fila_inputs = []
        for j in range(n):
            with cols[j]:
                val = st.number_input(f"A[{i+1},{j+1}]", value=0.0, format="%.2f", key=f"plu_a_{i}_{j}")
                fila_inputs.append(val)
        matriz_inputs.append(fila_inputs)
    
    # Vector b
    st.subheader("Vector b")
    b_inputs = []
    cols_b = st.columns(n)
    for i in range(n):
        with cols_b[i]:
            val = st.number_input(f"b[{i+1}]", value=0.0, format="%.2f", key=f"plu_b_{i}")
            b_inputs.append(val)
    
    if st.button("Resolver con PLU", use_container_width=True):
        A = np.array(matriz_inputs, dtype=float)
        b = np.array(b_inputs, dtype=float)
        
        df, info = factorizacion_plu(A, b)
        
        if info["ok"]:
            st.write("### Matrices Resultantes")
            c1, c2, c3 = st.columns(3)
            with c1: st.write("**P (Permutación)**"); st.write(info["P"])
            with c2: st.write("**L (Inferior)**"); st.write(info["L"])
            with c3: st.write("**U (Superior)**"); st.write(info["U"])
            
            st.subheader("Pasos de la Transformación (U)")
            st.dataframe(df, use_container_width=True)
            
            st.success(f"Solución x: {info['solucion']}")
        else:
            st.error(info["msg"])

def page_minimos_cuadrados():
    hero("Aproximación · Mínimos Cuadrados", "Encuentra la curva que mejor se ajusta a un conjunto de puntos.")

    # Entrada de datos
    col_input1, col_input2 = st.columns(2)
    x_str = col_input1.text_input("Valores de X (separados por coma)", "1, 2, 3, 4, 5")
    y_str = col_input2.text_input("Valores de Y (separados por coma)", "2.1, 3.9, 6.2, 8.1, 10.1")
    
    tipo = st.selectbox("Tipo de Ajuste", ["Lineal", "Polinomial", "No Lineal (Exponencial)"])
    grado = 2
    if tipo == "Polinomial":
        grado = st.slider("Grado del polinomio", 2, 6, 2)

    if st.button("Calcular Ajuste", use_container_width=True):
        try:
            x = [float(i) for i in x_str.split(",")]
            y = [float(i) for i in y_str.split(",")]
            
            if len(x) != len(y):
                st.error("X y Y deben tener la misma cantidad de datos.")
                return

            df, info = minimos_cuadrados(x, y, tipo, grado)
            
            if info["ok"]:
                st.success(f"**Modelo encontrado:** {info['modelo']}")
                st.info(f"**Coeficiente de determinación R²:** {info['r2']:.4f}")
                
                # Gráfica
                fig, ax = plt.subplots()
                ax.scatter(x, y, color="red", label="Datos Reales")
                
                # Curva suave para el ajuste
                x_smooth = np.linspace(min(x), max(x), 100)
                if tipo == "Lineal":
                    y_smooth = info['solucion'][0] + info['solucion'][1] * x_smooth
                elif tipo == "Polinomial":
                    y_smooth = np.polyval(info['solucion'][::-1], x_smooth)
                else:
                    y_smooth = info['solucion'][0] * np.exp(info['solucion'][1] * x_smooth)
                
                ax.plot(x_smooth, y_smooth, label="Curva de Ajuste")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                st.subheader("Análisis de Residuos")
                st.dataframe(df, use_container_width=True)
            else:
                st.error(info["msg"])
                
        except Exception as e:
            st.error(f"Error en los datos: {e}")
# =========================
# Catálogo de secciones y métodos 
# =========================
CATALOG = {
    "Ecuaciones de una variable": {
        "Bisección": page_biseccion,
        "Secante": page_secante,
        "Falsa Posición": page_falsa_posicion,
        "Newton": page_newton,
        "Punto Fijo": page_punto_fijo,
        "Müller": page_muller,
    },
    "Álgebra Lineal": {
        "Sistemas de Ecuaciones Lineales": page_pivoteos,
        "Factorizaciones": page_factorizacion_lu,
        "Factorización PLU": page_plu,
        "Factorización Cholesky": page_cholesky,
    },
    "Derivación": {
        "Derivación 2, 3, 5 puntos": lambda: not_ready("Derivación 2, 3, 5 puntos"),
        "Con h irregular": lambda: not_ready("Derivación con h irregular"),
        "Extrapolación": lambda: not_ready("Extrapolación (derivación)"),
    },
    "Integración": {
        "Reglas compuestas": lambda: not_ready("Reglas compuestas"),
        "Cuadratura adaptativa": lambda: not_ready("Cuadratura adaptativa"),
        "Cuadratura gaussiana": lambda: not_ready("Cuadratura gaussiana"),
        "Extrapolación de Romberg": lambda: not_ready("Romberg"),
        "Integración múltiple": lambda: not_ready("Integración múltiple"),
    },
    "Interpolación": {
        "Diferencias Divididas": lambda: not_ready("Diferencias Divididas"),
        "Neville": lambda: not_ready("Neville"),
        "Lagrange": lambda: not_ready("Lagrange"),
    },
    "EDOs": {
        "Método de Euler": lambda: not_ready("Euler"),
        "Método de Taylor de Orden Superior": lambda: not_ready("Taylor orden superior"),
        "Runge-Kutta 2, 3, 4": lambda: not_ready("Runge-Kutta 2/3/4"),
        "RKF": lambda: not_ready("RKF"),
        "Adams Bashforth": lambda: not_ready("Adams-Bashforth"),
        "Sistemas de ecuaciones": lambda: not_ready("Sistemas de EDOs"),
    },
    "Aproximaciones": {
        "Mínimos cuadrados": page_minimos_cuadrados,
    },
}


# =========================
# App principal
# =========================
st.set_page_config(page_title="Métodos Numéricos", layout="wide")
inject_css()

st.sidebar.title("Métodos Numéricos")
st.sidebar.caption("Seleccione una sección y un método.")

seccion = st.sidebar.radio("Sección", list(CATALOG.keys()))
metodo = st.sidebar.selectbox("Método", list(CATALOG[seccion].keys()))

# Header principal
hero("Métodos Numéricos", "Interfaz por categorías · Tablas de iteraciones · Resultados claros")

# Contenido
CATALOG[seccion][metodo]()

with st.sidebar.expander("Ayuda rápida", expanded=False):
    st.write("**Para ecuaciones:**")
    st.code("x**3 - x - 2\nsin(x) + x**2\nexp(x) - 5", language="text")
    st.write("Use ** para potencias. Ej: x**2 (no x^2).")
    st.write("")
    st.write("**Para sistemas lineales:**")
    st.write("Ingrese los coeficientes de la matriz A y el vector b.")
    st.write("Los pivoteos resolverán el sistema Ax = b.")