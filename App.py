import streamlit as st
from sympy import symbols, sympify, lambdify
from Bisection import bisection
from Secante import secante
from Muller import muller
from FalsaPosicion import falsa_posicion
from newton_raphson import newton
from punto_fijo import p_fijo

# Para manejarla como si fuera una función en el formato de matemáticas del cuaderno
def parse_function(expr_str):
    x = symbols('x')
    expr = sympify(expr_str, dict(x=x))
    f = lambdify(x, expr, 'numpy')
    return f

#Interfaz
st.set_page_config(page_title="Métodos Numéricos", layout="centered")
st.title("Métodos Numéricos de una Variable")

metodo = st.selectbox("Selecciona el método numérico", ["Método", "Bisección", "Secante", "Muller", "Falsa Posición", "Newton Raphson", "Punto fijo"])



if metodo == "Bisección":
    st.title("Método de la Bisección")
    expr_str = st.text_input("Introduce la ecuación f(x) =", "")
    tol_input = st.text_input("Error máximo (tolerancia)", "")
    itmax_input = st.text_input("Iteraciones máximas", "")
    col1, col2 = st.columns(2)
    with col1:
        a_input = st.text_input("Valor de a (límite inferior)", "")
    with col2:
        b_input = st.text_input("Valor de b (límite superior)", "")
    if st.button("Resolver"):
        try:
            # Conversión de los inputs
            tol = float(tol_input)
            itmax = int(itmax_input)
            a = float(a_input)
            b = float(b_input)

            # Formato de la ecuación
            f = parse_function(expr_str)

            # Resolución
            df, info = bisection(f, a, b, tol, itmax)

            if not info.get("ok", True):
                st.error(info["msg"])
            else:
                st.subheader("Tabla de Iteraciones")
                st.dataframe(df)
                st.success(f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}")
        except Exception as e:
            st.error(f"Ocurrió un error al resolver la ecuación: {e}")
            
if metodo == "Secante":
    st.title("Método de la Secante")
    expr_str = st.text_input("Introduce la ecuación f(x) =", "")
    tol_input = st.text_input("Error máximo (tolerancia)", "")
    itmax_input = st.text_input("Iteraciones máximas", "")
    col1, col2 = st.columns(2)
    with col1:
        a_input = st.text_input("Valor de a (límite inferior)", "")
    with col2:
        b_input = st.text_input("Valor de b (límite superior)", "")
    if st.button("Resolver"):
        try:
            # Conversión de los inputs
            tol = float(tol_input)
            itmax = int(itmax_input)
            a = float(a_input)
            b = float(b_input)

            # Formato de la ecuación
            f = parse_function(expr_str)

            # Resolución
            df, info = secante(f, a, b, tol, itmax)

            if not info.get("ok", True):
                st.error(info["msg"])
            else:
                st.subheader("Tabla de Iteraciones")
                st.dataframe(df)
                st.success(f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}")
        except Exception as e:
            st.error(f"Ocurrió un error al resolver la ecuación: {e}")

if metodo == "Muller":
    st.title("Método de Muller")
    expr_str = st.text_input("Introduce la ecuación f(x) =", "")
    p0str = st.text_input("Introduzca P0", "")
    p1str = st.text_input("Introduzca P1", "")
    p2str = st.text_input("Introduzca P2", "")
    tol_input = st.text_input("Error máximo (tolerancia)", "")
    itmax_input = st.text_input("Iteraciones máximas", "")
    if st.button("Resolver"):
        try:
            # Conversión de inputs
            p0 = float(p0str)
            p1 = float(p1str)
            p2 = float(p2str)
            tol = float(tol_input)
            itmax = int(itmax_input)

            f = parse_function(expr_str)

            # Resolución
            df, info = muller(f, p0, p1, p2, tol, itmax)

            if not info.get("ok", True):
                st.error(info["msg"])
            else:
                st.subheader("Tabla de Iteraciones")
                st.dataframe(df)
                st.success(f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}")
        except Exception as e:
            st.error(f"Ocurrió un error al resolver la ecuación: {e}")

if metodo == "Falsa Posición":
    st.title("Método de Falsa Posición")
    expr_str = st.text_input("Introduce la ecuación f(x) =", "")
    tol_input = st.text_input("Error máximo (tolerancia)", "")
    itmax_input = st.text_input("Iteraciones máximas", "")
    col1, col2 = st.columns(2)
    with col1:
        a_input = st.text_input("Valor de a (límite inferior)", "")
    with col2:
        b_input = st.text_input("Valor de b (límite superior)", "")
    if st.button("Resolver"):
        try:
            # Conversión de los inputs
            tol = float(tol_input)
            itmax = int(itmax_input)
            a = float(a_input)
            b = float(b_input)

            # Formato de la ecuación
            f = parse_function(expr_str)

            # Resolución
            df, info = falsa_posicion(f, a, b, tol, itmax)

            if not info.get("ok", True):
                st.error(info["msg"])
            else:
                st.subheader("Tabla de Iteraciones")
                st.dataframe(df)
                st.success(f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}")
        except Exception as e:
            st.error(f"Ocurrió un error al resolver la ecuación: {e}")
    
if metodo == "Newton Raphson":
    st.title("Método de Newton Raphson")
    expr_str = st.text_input("Introduce la ecuación f(x) = ", "")
    derivada_str = st. text_input("Introduce la derivada de f(x = )", "")
    tol_input = st.text_input("Error máximo (tolerancia)", "")
    itmax_input = st.text_input("Iteraciones máximas", "")
    a_input = st.text_input("Valor de a (límite inferior)", "")
    if st.button("Resolver"):
        try:
            # Conversión de los inputs
            tol = float(tol_input)
            itmax = int(itmax_input)
            a = float(a_input)

            # Formato de la ecuación
            f = parse_function(expr_str)
            f_der = parse_function(derivada_str)

            df, info = newton(f, f_der, a, itmax, tol)

            if not info.get("ok", True):
                st.error(info["msg"])
            else:
                st.subheader("Tabla de Iteraciones")
                st.dataframe(df)
                st.success(f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}")
        except Exception as e:
            st.error(f"Ocurrió un error al resolver la ecuación: {e}")

if metodo == "Punto fijo":
    st.title("Método de Punto Fijo")
    expr_str = st.text_input("Introduce la ecuación f(x) = ", "")
    tol_input = st.text_input("Error máximo (tolerancia)", "")
    itmax_input = st.text_input("Iteraciones máximas", "")
    a_input = st.text_input("Valor de a (límite inferior)", "")
    if st.button("Resolver"):
        try:
            # Conversión de los inputs
            tol = float(tol_input)
            itmax = int(itmax_input)
            a = float(a_input)

            # Formato de la ecuación
            f = parse_function(expr_str)

            df, info = p_fijo(f, a, itmax, tol)

            if not info.get("ok", True):
                st.error(info["msg"])
            else:
                st.subheader("Tabla de Iteraciones")
                st.dataframe(df)
                st.success(f"Raíz aproximada: {info.get('raiz')} | Iteraciones: {info.get('iter')} | Error final: {info.get('error')}")
        except Exception as e:
            st.error(f"Ocurrió un error al resolver la ecuación: {e}")
