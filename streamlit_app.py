import gdown
import streamlit as st

from studcamp_yandex_hse.models import RakeBasedTagger

st.set_page_config(
    page_title="Text Tagger",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def main():
    file_url = "https://drive.google.com/file/d/1EppNyj2zfwHuSnZWTRtAPfDGJArSq-m8/view?usp=sharing"
    destination_path = "cc.ru.300.bin"
    gdown.download(file_url, destination_path, quiet=False)

    st.title("Hello World!")

    input_text = st.text_input("Your text is here")
    if st.button("Submit"):
        if type(input_text) is str:
            st.write(RakeBasedTagger().extract(text, 5))
            st.balloons()
        else:
            st.error("You need to input a text in the field above")


if __name__ == "__main__":
    main()
