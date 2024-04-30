import gdown
import streamlit as st

from studcamp_yandex_hse.models import (
    AttentionBasedTagger,
    BartBasedTagger,
    DBSCANFaissTagger,
    RakeBasedTagger,
    Rut5BasedTagger,
)

st.set_page_config(
    page_title="Text Tagger",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_data():
    url_bin = "https://drive.google.com/uc?id=1EppNyj2zfwHuSnZWTRtAPfDGJArSq-m8"
    url_weights = "https://drive.google.com/drive/u/0/folders/11cnQXsSJUteyUfuv4F7NStyR7PVHWEzy"
    output = "./"
    gdown.download(url_bin, output)
    gdown.download_folder(url_weights)


def main():
    st.title("🔮 Inference section")

    load_data()

    input_text = st.text_area("", placeholder="Your text is here")
    if st.button("Proceed"):
        if type(input_text) is str:
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                with st.expander("RakeBased Tags"):
                    st.write(",".join(RakeBasedTagger().extract(input_text, 5)))
            with col2:
                with st.expander("BartBased Tags"):
                    st.write(",".join(BartBasedTagger().extract(text, 5)))
            with col3:
                with st.expander("ClusterizedBased Tags"):
                    st.write(",".join(DBSCANFaissTagger().extract(text, 5)))
            with col4:
                with st.expander("AttentionBased Tags"):
                    st.write(",".join(AttentionBasedTagger().extract(text, 5)))
            with col5:
                with st.expander("RuT5Based Tags"):
                    st.write(",".join(Rut5BasedTagger().extract(input_text)))

            st.balloons()

        else:
            st.error("You need to input a text in the field above")


if __name__ == "__main__":
    main()
