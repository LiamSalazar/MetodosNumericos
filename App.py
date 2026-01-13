# App.py
import streamlit as st
from sympy import symbols, sympify, lambdify
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sympy import pi, E 
import pandas as pd

# Ecuaciones de una variable
from Bisection import bisection
from Secante import secante
from Muller import muller
from FalsaPosicion import falsa_posicion
from newton_raphson import newton
from punto_fijo import p_fijo

# Integración
from ReglasCompuestas import trapecio_compuesto, simpson_13_compuesto, simpson_38_compuesto
from CuadraturaAdaptativa import simpson_adaptativo
from Gaussiana import gauss_legendre
from Romberg import romberg
from SimpsonDoble import simpson_doble
from GaussDoble import gauss_doble
from GaussTriple import gauss_triple

# EDOs
from RKF import rkf45
from edos_adams_bashforth import adams_bashforth
from RK4_Sistemas import rk4_sistemas


# =========================
# Utilidades
# =========================
def parse_edo_system(funcs_str_list):
    """
    Recibe lista de strings [f1, f2, ..., fn] y retorna F(x, y_vec)->np.array.
    Soporta nombres de variables:
      x (o t)
      y1,y2,...,yn
      y,z,w,u,v (según n)
    """
    import numpy as np
    from sympy import symbols, sympify, lambdify, pi, E

    x = symbols("x")

    # alias para variables comunes en sistemas
    aliases = ["y", "z", "w", "u", "v", "s", "r", "p", "q"]
    n = len(funcs_str_list)

    y_syms = symbols(" ".join([f"y{i+1}" for i in range(n)]))

    # locals: x, t, y1..yn, y/z/w/u/v...
    loc = {"x": x, "t": x, "pi": pi, "e": E, "E": E}
    for i in range(n):
        loc[f"y{i+1}"] = y_syms[i]
        if i < len(aliases):
            loc[aliases[i]] = y_syms[i]

    exprs = []
    for s in funcs_str_list:
        exprs.append(sympify(s, locals=loc))

    f_lambdas = [lambdify((x, *y_syms), e, "numpy") for e in exprs]

    def F(xval, yvec):
        yvec = np.array(yvec, dtype=float).reshape(-1)
        args = (xval, *yvec.tolist())
        out = [fi(*args) for fi in f_lambdas]
        return np.array(out, dtype=float).reshape(-1)

    return F


def parse_edo_function(expr_str: str):
    """Convierte 'y - x**2 + 1' -> f(x,y) evaluable con numpy."""
    x, y = symbols("x y")
    expr = sympify(expr_str, dict(x=x, y=y, pi=pi, e=E, E=E))
    f = lambdify((x, y), expr, "numpy")
    return f


def parse_function_xy(expr_str: str):
    x, y = symbols("x y")
    expr = sympify(expr_str, dict(x=x, y=y))
    f = lambdify((x, y), expr, "numpy")
    return f

def parse_function_xyz(expr_str: str):
    x, y, z = symbols("x y z")
    expr = sympify(expr_str, dict(x=x, y=y, z=z))
    f = lambdify((x, y, z), expr, "numpy")
    return f

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

# =========================
# Ecuaciones de una variable
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
# Integración
# =========================

def page_reglas_compuestas():
    hero("Integración · Reglas compuestas", "Trapecio compuesto, Simpson 1/3 compuesto y Simpson 3/8 compuesto.")

    expr_str = st.text_input("Introduce la función f(x) =", "")

    col1, col2, col3 = st.columns(3)
    a_input = col1.text_input("a (límite inferior)", "")
    b_input = col2.text_input("b (límite superior)", "")
    n_input = col3.text_input("n (subintervalos)", "")

    regla = st.selectbox("Regla", ["Trapecio compuesto", "Simpson 1/3 compuesto", "Simpson 3/8 compuesto"])

    if st.button("Resolver", use_container_width=True):
        try:
            a = float(a_input)
            b = float(b_input)
            n = int(n_input)
            f = parse_function(expr_str)

            if regla == "Trapecio compuesto":
                df, info = trapecio_compuesto(f, a, b, n)
            elif regla == "Simpson 1/3 compuesto":
                df, info = simpson_13_compuesto(f, a, b, n)
            else:
                df, info = simpson_38_compuesto(f, a, b, n)

            if not info.get("ok", True):
                st.error(info["msg"])
                return

            st.subheader("Tabla de cálculo")
            st.dataframe(df, use_container_width=True)

            st.success(f"Integral aproximada: {info.get('I')} | h: {info.get('h')} | n: {info.get('n')}")
            
            # =========================
            # Gráficas de apoyo
            # =========================
            st.subheader("Gráfica de f(x) en [a, b] y nodos de integración")

            x_nodes = info["x"]
            fx_nodes = info["fx"]

            # Curva suave para visualizar la función
            xx = np.linspace(a, b, 600)
            yy = f(xx)

            fig, ax = plt.subplots()
            ax.plot(xx, yy, label="f(x)")
            ax.scatter(x_nodes, fx_nodes, label="Nodos x_i")
            ax.axhline(0, linewidth=1)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.grid(True, alpha=0.25)
            ax.legend()
            st.pyplot(fig)

            # Visualización extra solo para trapecio: trapecios rellenos
            if regla == "Trapecio compuesto":
                st.subheader("Visualización del Trapecio compuesto")

                fig2, ax2 = plt.subplots()
                ax2.plot(xx, yy, label="f(x)")
                ax2.scatter(x_nodes, fx_nodes, label="Nodos x_i")

                for i in range(len(x_nodes) - 1):
                    x0, x1 = x_nodes[i], x_nodes[i + 1]
                    y0, y1 = fx_nodes[i], fx_nodes[i + 1]
                    ax2.fill([x0, x0, x1, x1], [0, y0, y1, 0], alpha=0.2)

                ax2.axhline(0, linewidth=1)
                ax2.set_xlabel("x")
                ax2.set_ylabel("f(x)")
                ax2.grid(True, alpha=0.25)
                ax2.legend()
                st.pyplot(fig2)


        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")


def page_cuadratura_adaptativa():
    hero("Integración · Cuadratura adaptativa", "Simpson adaptativo con subdivisión automática según tolerancia.")

    expr_str = st.text_input("Introduce la función f(x) =", "")

    col1, col2, col3 = st.columns(3)
    a_input = col1.text_input("a (límite inferior)", "")
    b_input = col2.text_input("b (límite superior)", "")
    tol_input = col3.text_input("Tolerancia (error máximo)", "")

    max_depth = st.number_input("Profundidad máxima de subdivisión", min_value=1, max_value=50, value=20, step=1)

    if st.button("Resolver", use_container_width=True):
        try:
            a = float(a_input)
            b = float(b_input)
            tol = float(tol_input)
            f = parse_function(expr_str)

            df, info = simpson_adaptativo(f, a, b, tol, int(max_depth))
            if not info.get("ok", True):
                st.error(info["msg"])
                return

            st.subheader("Tabla de subintervalos aceptados")
            st.dataframe(df, use_container_width=True)

            st.success(
                f"Integral aproximada: {info.get('I')} | Subintervalos: {info.get('subintervalos')} | "
                f"Profundidad máx.: {info.get('profundidad_max')} | tol: {info.get('tol')}"
            )
            # =========================
            # Gráficas 
            # =========================
            st.subheader("Gráfica de f(x) y partición adaptativa")

            # Curva suave de la función
            xx = np.linspace(a, b, 800)
            yy = f(xx)

            fig, ax = plt.subplots()

            # f(x)
            ax.plot(xx, yy, label="f(x)", linewidth=2)

            # Eje horizontal
            ax.axhline(0, linewidth=1)

            # Líneas verticales de la partición adaptativa
            # df contiene los subintervalos aceptados
            bounds = np.unique(
                np.concatenate([df["a"].to_numpy(), df["b"].to_numpy()])
            )
            bounds.sort()

            for xline in bounds:
                ax.axvline(float(xline), linestyle="--", linewidth=0.8, alpha=0.4)

            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title("Cuadratura adaptativa: refinamiento automático del intervalo")
            ax.grid(True, alpha=0.25)
            ax.legend()

            st.pyplot(fig)



        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")

def page_cuadratura_gaussiana():
    hero("Integración · Cuadratura gaussiana", "Gauss-Legendre (2 a 6 puntos) en forma compuesta.")

    expr_str = st.text_input("Introduce la función f(x) =", "")
    col1, col2, col3 = st.columns(3)
    a_input = col1.text_input("a (límite inferior)", "")
    b_input = col2.text_input("b (límite superior)", "")
    m_input = col3.text_input("m (subintervalos)", "")

    order = st.selectbox("Orden (puntos de Gauss)", [2, 3, 4, 5, 6], index=1)

    if st.button("Resolver", use_container_width=True):
        try:
            a = float(a_input)
            b = float(b_input)
            m = int(m_input)
            f = parse_function(expr_str)

            df, info = gauss_legendre(f, a, b, m, order)
            if not info.get("ok", True):
                st.error(info["msg"])
                return

            st.subheader("Tabla de cálculo")
            st.dataframe(df, use_container_width=True)

            st.success(
                f"Integral aproximada: {info.get('I')} | m: {info.get('m')} | order: {info.get('order')} | h: {info.get('h')}"
            )

            # =========================
            # Gráfica única (obligatoria)
            # f(x) + nodos de Gauss
            # =========================
            import numpy as np
            import matplotlib.pyplot as plt

            st.subheader("Gráfica de f(x) y nodos de Gauss")

            aa = min(a, b)
            bb = max(a, b)
            xx = np.linspace(aa, bb, 900)
            yy = f(xx)

            fig, ax = plt.subplots()
            ax.plot(xx, yy, label="f(x)", linewidth=2)
            ax.axhline(0, linewidth=1)

            # Límites de subintervalos
            bounds = np.linspace(aa, bb, info["m"] + 1)
            for xline in bounds:
                ax.axvline(float(xline), linestyle="--", linewidth=0.8, alpha=0.35)

            # Nodos (x_ij)
            x_nodes = df["x_ij"].to_numpy()
            y_nodes = df["f(x_ij)"].to_numpy()
            ax.scatter(x_nodes, y_nodes, label="Nodos de Gauss", s=25)

            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title("Cuadratura gaussiana: nodos de Gauss-Legendre")
            ax.grid(True, alpha=0.25)
            ax.legend()

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")

def page_romberg():
    hero("Integración · Extrapolación de Romberg", "Trapecio + extrapolación Richardson (tabla triangular).")

    expr_str = st.text_input("Introduce la función f(x) =", "")
    col1, col2, col3 = st.columns(3)
    a_input = col1.text_input("a (límite inferior)", "")
    b_input = col2.text_input("b (límite superior)", "")
    tol_input = col3.text_input("Tolerancia (tol)", "")

    max_level_input = st.text_input("Niveles máximos (max_level)", "6")

    if st.button("Resolver", use_container_width=True):
        try:
            a = float(a_input)
            b = float(b_input)
            tol = float(tol_input)
            max_level = int(max_level_input)

            f = parse_function(expr_str)

            df, info = romberg(f, a, b, max_level, tol)
            if not info.get("ok", True):
                st.error(info["msg"])
                return

            # Tabla triangular: pivot k vs j
            tri = df.pivot(index="k", columns="j", values="R_kj").sort_index()
            st.subheader("Tabla de Romberg (R[k,j])")
            st.dataframe(tri, use_container_width=True)

            st.success(
                f"Integral aproximada: {info.get('I')} | k_final: {info.get('k_final')} | n_final: {info.get('n_final')} | converged: {info.get('converged')}"
            )

            # =========================
            # Gráfica única útil
            # f(x) + partición del trapecio en el nivel final (n_final)
            # =========================
            import numpy as np
            import matplotlib.pyplot as plt

            st.subheader("Gráfica de f(x) y partición final usada por trapecio (nivel final)")

            aa = min(a, b)
            bb = max(a, b)
            xx = np.linspace(aa, bb, 900)
            yy = f(xx)

            fig, ax = plt.subplots()
            ax.plot(xx, yy, label="f(x)", linewidth=2)
            ax.axhline(0, linewidth=1)

            # Líneas verticales del trapecio final (n_final subintervalos)
            n_final = int(info["n_final"])
            bounds = np.linspace(aa, bb, n_final + 1)
            for xline in bounds:
                ax.axvline(float(xline), linestyle="--", linewidth=0.8, alpha=0.30)

            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title("Romberg: malla final (trapecio) que alimenta la extrapolación")
            ax.grid(True, alpha=0.25)
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")


def page_simpson_doble():
    hero("Integración múltiple · Simpson doble", "Simpson 1/3 compuesto en x y en y (producto tensorial).")

    expr_str = st.text_input("Introduce la función f(x,y) =", "x**2 + y**2")
    col1, col2, col3, col4 = st.columns(4)
    ax_in = col1.text_input("ax", "0")
    bx_in = col2.text_input("bx", "1")
    ay_in = col3.text_input("ay", "0")
    by_in = col4.text_input("by", "1")

    col5, col6 = st.columns(2)
    nx_in = col5.text_input("nx (par)", "10")
    ny_in = col6.text_input("ny (par)", "10")

    if st.button("Resolver", use_container_width=True):
        try:
            ax = float(ax_in); bx = float(bx_in)
            ay = float(ay_in); by = float(by_in)
            nx = int(nx_in); ny = int(ny_in)

            f = parse_function_xy(expr_str)
            df, info = simpson_doble(f, ax, bx, ay, by, nx, ny)
            if not info.get("ok", True):
                st.error(info["msg"]); return

            st.subheader("Tabla de cálculo")
            st.dataframe(df, use_container_width=True)

            st.success(f"Integral aproximada: {info['I']} | hx: {info['hx']} | hy: {info['hy']} | nx: {info['nx']} | ny: {info['ny']}")

            # Gráfica 3D (superficie)
            st.subheader("Gráfica 3D de f(x,y) y nodos de la malla")
            X = np.linspace(ax, bx, 60)
            Y = np.linspace(ay, by, 60)
            XX, YY = np.meshgrid(X, Y)
            ZZ = f(XX, YY)

            # nodos de Simpson (malla)
            x_nodes = np.linspace(ax, bx, nx + 1)
            y_nodes = np.linspace(ay, by, ny + 1)
            XN, YN = np.meshgrid(x_nodes, y_nodes)
            ZN = f(XN, YN)

            fig = plt.figure()
            ax3d = fig.add_subplot(111, projection="3d")
            ax3d.plot_surface(XX, YY, ZZ, alpha=0.85)
            ax3d.scatter(XN.flatten(), YN.flatten(), ZN.flatten(), s=12, alpha=0.9, label="Nodos")
            ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("f(x,y)")
            ax3d.set_title("Simpson doble: superficie y nodos")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

def page_gauss_doble():
    hero("Integración múltiple · Gauss doble", "Gauss-Legendre (2 a 6 puntos) en forma compuesta.")

    expr_str = st.text_input("Introduce la función f(x,y) =", "x**2 + y**2")
    col1, col2, col3, col4 = st.columns(4)
    ax_in = col1.text_input("ax", "0")
    bx_in = col2.text_input("bx", "1")
    ay_in = col3.text_input("ay", "0")
    by_in = col4.text_input("by", "1")

    col5, col6, col7 = st.columns(3)
    mx_in = col5.text_input("mx (subintervalos en x)", "1")
    my_in = col6.text_input("my (subintervalos en y)", "1")
    orden = col7.selectbox("Orden n (puntos de Gauss)", [2, 3, 4, 5, 6], index=1)

    if st.button("Resolver", use_container_width=True):
        try:
            axv = float(ax_in); bxv = float(bx_in)
            ayv = float(ay_in); byv = float(by_in)
            mx = int(mx_in); my = int(my_in)

            f = parse_function_xy(expr_str)
            df, info = gauss_doble(f, axv, bxv, ayv, byv, mx, my, int(orden))
            if not info.get("ok", True):
                st.error(info["msg"]); return

            st.subheader("Tabla de cálculo")
            st.dataframe(df, use_container_width=True)
            st.success(f"Integral aproximada: {info['I']} | mx:{info['mx']} my:{info['my']} | orden:{info['orden']}")

            st.subheader("Gráfica 3D: superficie y nodos de Gauss")
            X = np.linspace(axv, bxv, 70)
            Y = np.linspace(ayv, byv, 70)
            XX, YY = np.meshgrid(X, Y)
            ZZ = f(XX, YY)

            fig = plt.figure()
            ax3d = fig.add_subplot(111, projection="3d")
            ax3d.plot_surface(XX, YY, ZZ, alpha=0.85)

            # nodos (tomamos los x_ij,y_ij de la tabla)
            ax3d.scatter(df["x_ij"].to_numpy(), df["y_ij"].to_numpy(), df["f(x_ij,y_ij)"].to_numpy(),
                         s=18, alpha=0.95, label="Nodos de Gauss")

            ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("f(x,y)")
            ax3d.set_title("Gauss doble: superficie y nodos")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

def page_gauss_triple():
    hero("Integración múltiple · Gauss triple", "Gauss-Legendre triple en forma compuesta. Visualización por cortes en z.")

    expr_str = st.text_input("Introduce la función f(x,y,z) =", "x**2 + y**2 + z**2")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ax_in = c1.text_input("ax", "0")
    bx_in = c2.text_input("bx", "1")
    ay_in = c3.text_input("ay", "0")
    by_in = c4.text_input("by", "1")
    az_in = c5.text_input("az", "0")
    bz_in = c6.text_input("bz", "1")

    c7, c8, c9, c10 = st.columns(4)
    mx_in = c7.text_input("mx", "1")
    my_in = c8.text_input("my", "1")
    mz_in = c9.text_input("mz", "1")
    orden = c10.selectbox("Orden n", [2, 3, 4, 5, 6], index=1)

    if st.button("Resolver", use_container_width=True):
        try:
            axv = float(ax_in); bxv = float(bx_in)
            ayv = float(ay_in); byv = float(by_in)
            azv = float(az_in); bzv = float(bz_in)
            mx = int(mx_in); my = int(my_in); mz = int(mz_in)

            f = parse_function_xyz(expr_str)
            df, info = gauss_triple(f, axv, bxv, ayv, byv, azv, bzv, mx, my, mz, int(orden))
            if not info.get("ok", True):
                st.error(info["msg"]); return

            st.subheader("Tabla de cálculo (muestra)")
            st.dataframe(df.head(300), use_container_width=True)
            st.caption("Se muestra una porción de la tabla si es muy grande.")
            st.success(f"Integral aproximada: {info['I']} | mx:{mx} my:{my} mz:{mz} | orden:{info['orden']}")

            # Gráfica: cortes en z
            st.subheader("Gráfica 3D por cortes en z (z=a, z=medio, z=b)")

            z_slices = [azv, 0.5*(azv+bzv), bzv]
            X = np.linspace(axv, bxv, 70)
            Y = np.linspace(ayv, byv, 70)
            XX, YY = np.meshgrid(X, Y)

            fig = plt.figure()
            ax3d = fig.add_subplot(111, projection="3d")

            for z0 in z_slices:
                ZZ = f(XX, YY, z0)
                ax3d.plot_surface(XX, YY, ZZ, alpha=0.55)

            # nodos: solo los que caen cerca del corte (para no saturar)
            zvals = df["z"].to_numpy()
            tolz = (bzv - azv) / max(50, mz*int(orden)*5)
            mask = np.zeros_like(zvals, dtype=bool)
            for z0 in z_slices:
                mask |= np.abs(zvals - z0) <= tolz

            dfm = df[mask]
            if len(dfm) > 0:
                ax3d.scatter(dfm["x"].to_numpy(), dfm["y"].to_numpy(), dfm["f(x,y,z)"].to_numpy(),
                             s=14, alpha=0.9, label="Nodos (cercanos a cortes)")

            ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("f(x,y,z)")
            ax3d.set_title("Gauss triple: superficies por cortes en z")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

# =========================
# EDOs
# =========================

def page_rkf():
    hero("EDOs · RKF45", "Runge–Kutta–Fehlberg 4(5) con paso adaptativo y control de error.")

    expr_str = st.text_input("Introduce la EDO y' = f(x, y) =", "")

    c1, c2, c3 = st.columns(3)
    x0_input = c1.text_input("x0", "")
    y0_input = c2.text_input("y0", "")
    xf_input = c3.text_input("xf", "")

    c4, c5, c6 = st.columns(3)
    h0_input = c4.text_input("h0 (paso inicial)", "")
    tol_input = c5.text_input("Tolerancia (error por paso)", "")
    itmax_input = c6.text_input("Iteraciones máximas", "")

    with st.expander("Opcional: límites de paso", expanded=False):
        d1, d2 = st.columns(2)
        hmin_input = d1.text_input("h_min", "1e-12")
        hmax_input = d2.text_input("h_max", "1.0")

    if st.button("Resolver", use_container_width=True):
        try:
            x0 = float(x0_input)
            y0 = float(y0_input)
            xf = float(xf_input)
            h0 = float(h0_input)
            tol = float(tol_input)
            itmax = int(itmax_input)
            h_min = float(hmin_input)
            h_max = float(hmax_input)

            f = parse_edo_function(expr_str)

            df, info = rkf45(f, x0, y0, xf, h0, tol, itmax, h_min=h_min, h_max=h_max)

            if not info.get("ok", True):
                st.error(info["msg"])
                st.dataframe(df, use_container_width=True)
                return

            st.subheader("Tabla de pasos")
            st.dataframe(df, use_container_width=True)

            st.success(
                f"y({info.get('x_final')}) ≈ {info.get('y_final')} | "
                f"accepted: {info.get('accepted')} | rejected: {info.get('rejected')} | steps: {info.get('steps')}"
            )

            # Gráfica única (obligatoria)
            xs = info["xs"]
            ys = info["ys"]

            st.subheader("Gráfica y(x)")
            fig, ax = plt.subplots()
            ax.plot(xs, ys, label="y(x)")
            ax.scatter(xs, ys, s=18, label="pasos aceptados")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.25)
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ocurrió un error al resolver: {e}")


def ui_adams_bashforth():
    st.header("EDOs · Adams–Bashforth (explícito)")

    expr = st.text_input("Introduce la EDO y' = f(x,y) =", value="y")
    x0  = st.number_input("x0", value=0.0, format="%.12f")
    y0  = st.number_input("y0", value=1.0, format="%.12f")
    xf  = st.number_input("xf", value=1.0, format="%.12f")
    h   = st.number_input("h (paso fijo)", value=0.1, format="%.12f", min_value=1e-12)

    order = st.selectbox("Orden AB", options=[2,3,4], index=2)

    if st.button("Resolver"):
        f = parse_edo_function(expr)

        xs, ys, info, steps = adams_bashforth(f, x0, y0, xf, h, order=order)

        df = pd.DataFrame(steps)
        st.subheader("Tabla de pasos")
        st.dataframe(df, use_container_width=True)

        st.success(f"y({info['xf']}) ≈ {info['y_final']:.15g} | h={info['h']:.6g} | n={info['n_steps']} | orden=AB{info['order']}")

        # Gráfica única (útil): y(x) + marca de puntos
        st.subheader("Gráfica y(x) (paso fijo)")
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label="y(x)")
        ax.scatter(xs, ys, s=18, label="nodos", zorder=3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)
        ax.legend()
        st.pyplot(fig)
        
def page_rk4_sistemas():
    hero("EDOs · Sistemas · Runge-Kutta 4", "RK4 con paso fijo para sistemas y' = f(x, y_vec).")

    n_input = st.text_input("Número de ecuaciones (n)", "2")
    try:
        n = int(n_input)
        if n < 2:
            st.error("n debe ser >= 2")
            return
        if n > 6:
            st.warning("Recomendación: n <= 6 para visualización clara.")
    except:
        st.error("n debe ser un entero.")
        return

    st.subheader("Define el sistema (una ecuación por componente)")
    funcs = []
    for i in range(n):
        default = "0"
        if n == 2 and i == 0:
            default = "z"  # y1' = y2
        if n == 2 and i == 1:
            default = "-y" # y2' = -y1
        funcs.append(st.text_input(f"f{i+1}(x, y) = y{i+1}' =", value=default))

    st.subheader("Condiciones iniciales")
    c1, c2, c3 = st.columns(3)
    x0_input = c1.text_input("x0", "0")
    xf_input = c2.text_input("xf", "10")
    h_input  = c3.text_input("h (paso)", "0.1")

    y0_vec = []
    cols = st.columns(min(n, 4))
    for i in range(n):
        col = cols[i % len(cols)]
        y0_vec.append(col.text_input(f"y{i+1}(x0)", "0"))

    itmax_input = st.text_input("Iteraciones máximas", "200000")

    if st.button("Resolver", use_container_width=True):
        try:
            x0 = float(x0_input)
            xf = float(xf_input)
            h = float(h_input)
            itmax = int(itmax_input)
            y0 = [float(v) for v in y0_vec]

            F = parse_edo_system(funcs)
            df, info = rk4_sistemas(F, x0, y0, xf, h, itmax=itmax)

            if not info.get("ok", True):
                st.error(info.get("msg", "Error"))
                if len(df) > 0:
                    st.dataframe(df, use_container_width=True)
                return

            st.subheader("Tabla de pasos")
            st.dataframe(df, use_container_width=True)

            st.success(
                f"x_final: {info['x_final']} | "
                f"y_final: {info['y_final']} | "
                f"h: {info['h']} | pasos: {info['n_steps']}"
            )

            # Gráfica útil (obligatoria): y_i vs x
            st.subheader("Gráfica de componentes y_i(x)")
            xs = info["xs"]
            Ys = info["Ys"]

            fig, ax = plt.subplots()
            for i in range(info["nvar"]):
                ax.plot(xs, Ys[:, i], label=f"y{i+1}(x)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.25)
            ax.legend()
            st.pyplot(fig)

            # Gráfica adicional útil para n=2: plano fase (opcional)
            if info["nvar"] == 2:
                st.subheader("Plano fase (y2 vs y1)")
                fig2, ax2 = plt.subplots()
                ax2.plot(Ys[:, 0], Ys[:, 1])
                ax2.set_xlabel("y1")
                ax2.set_ylabel("y2")
                ax2.grid(True, alpha=0.25)
                st.pyplot(fig2)

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
        "Reglas compuestas": page_reglas_compuestas,
        "Cuadratura adaptativa": page_cuadratura_adaptativa,
        "Cuadratura gaussiana": page_cuadratura_gaussiana,
        "Extrapolación de Romberg": page_romberg,
    },
    "Integración múltiple": {
    "Integral doble de Simpson": page_simpson_doble,
    "Integral doble gaussiana": page_gauss_doble,
    "Integral triple gaussiana": page_gauss_triple,
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
        "RKF": page_rkf,
        "Adams–Bashforth": ui_adams_bashforth,
        "Sistemas de ecuaciones": page_rk4_sistemas,
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
hero("Métodos Numéricos", "Interfaz por categorías · Tablas de iteraciones · Resultados claros")

# Contenido
CATALOG[seccion][metodo]()

with st.sidebar.expander("Ayuda rápida", expanded=False):
    st.write("Formato recomendado para funciones:")
    st.code("x**3 - x - 2\nsin(x) + x**2\nexp(x) - 5", language="text")
    st.write("Use ** para potencias. Ej: x**2 (no x^2).")
