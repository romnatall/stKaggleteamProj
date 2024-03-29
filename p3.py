import romautils
import streamlit as st



def main():
    st.title("Классификация клеток крови")

    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)
        st.write("")
        d = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}
        predicted_class, confidence = romautils.classify_image(uploaded_file)
        st.write(f"предсказанный класс: {d[predicted_class]}")

