# App.py
import streamlit as st
from sympy import symbols, sympify, lambdify

# ==== Importa SOLO lo que ya existe en tu proyecto ====
from Bisection import bisection
from Secante import secante
from Muller import muller
from FalsaPosicion import falsa_posicion
from newton_raphson import newton
from punto_fijo import p_fijo


# =========================
# Utilidades
# =========================
def parse_function(expr_str: str):
    """Convierte 'x**2 - 2' -> función evaluable con numpy."""
    x = symbols("x")
    expr = sympify(expr_str, dict(x=x))
    f = lambdify(x, expr, "numpy")
    return f


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
            st.success(
                f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}"
            )
        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")


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
    "Álgebra Lineal": {
        "Pivoteos para SEL": lambda: not_ready("Pivoteos para SEL"),
        "Factorizaciones": lambda: not_ready("Factorizaciones"),
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
        "Mínimos cuadrados": lambda: not_ready("Mínimos cuadrados"),
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
hero("Suite de Métodos Numéricos", "Interfaz por categorías · Tablas de iteraciones · Resultados claros")

# Contenido
CATALOG[seccion][metodo]()

with st.sidebar.expander("Ayuda rápida", expanded=False):
    st.write("Formato recomendado para funciones:")
    st.code("x**3 - x - 2\nsin(x) + x**2\nexp(x) - 5", language="text")
    st.write("Use ** para potencias. Ej: x**2 (no x^2).")
