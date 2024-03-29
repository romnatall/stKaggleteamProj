import streamlit as st
import p1,p2,p3
import romautils
from romautils import Model,MyModel
import streamlit as st

from model import MyResNet


# Настройки страницы
st.set_page_config(page_title="прожект")


def page1():
    p1.main()

def page2():
    p2.main()

def page3(): 
    p3.main()

def main():
    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Перейти к:", ("О проекте", "класификация спорта", "классификация клеток"))

    if page == "О проекте":
        page1()
    elif page == "класификация спорта":
        page2()
    elif page == "классификация клеток":
        page3()

if __name__ == "__main__":
    main()


