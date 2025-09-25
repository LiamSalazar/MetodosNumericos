import streamlit as st
from sympy import symbols, sympify, lambdify
from Bisection import bisection
from Secante import secante
from Muller import muller

# Para manejarla como si fuera una función en el formato de matemáticas del cuaderno
def parse_function(expr_str):
    x = symbols('x')
    expr = sympify(expr_str, dict(x=x))
    f = lambdify(x, expr, 'numpy')
    return f

#Interfaz
st.set_page_config(page_title="Métodos Numéricos", layout="centered")
st.title("Métodos Numéricos de una Variable")

metodo = st.selectbox("Selecciona el método numérico", ["Método", "Bisección", "Secante", "Muller"])



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