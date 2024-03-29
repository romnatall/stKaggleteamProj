import streamlit as st


def main():
    st.title("О проекте")
    st.write("Проект по классификации изображений")
    st.image("pngs/output.png", caption="Предсказанный класс", use_column_width=True)