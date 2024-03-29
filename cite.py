import streamlit as st

import pages.sportclassifier as sportclassifier
import pages.cellclassifier as cellclassifier
import streamlit as st

import pages.sportclassifier as sportclassifier
import pages.cellclassifier as cellclassifier


# Настройки страницы
st.set_page_config(page_title="прожект")


st.title("О проекте")
st.write("Проект по классификации изображений")
st.image("pngs/output.png", caption="Предсказанный класс", use_column_width=True)
