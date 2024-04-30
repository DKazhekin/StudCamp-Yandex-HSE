import chakin
import gdown
import streamlit as st

from studcamp_yandex_hse.models import (
    AttentionBasedTagger,
    BartBasedTagger,
    DBSCANFaissTagger,
    RakeBasedTagger,
    Rut5BasedTagger,
)
from studcamp_yandex_hse.processing.embedder import FastTextEmbedder

st.set_page_config(
    page_title="Text Tagger",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner="Loading data")
def load_data():
    chakin.download(number=9, save_dir="./")
    url_weights = "https://drive.google.com/drive/u/0/folders/11cnQXsSJUteyUfuv4F7NStyR7PVHWEzy"
    gdown.download_folder(url_weights)


@st.cache_resource(show_spinner="Loading Embeddings")
def load_ft_emb_model():
    emb_model = FastTextEmbedder()
    return emb_model


@st.cache_resource(show_spinner="Loading Rake")
def load_rake(_emb_model):
    rake_model = RakeBasedTagger(_emb_model)
    return rake_model


@st.cache_resource(show_spinner="Loading RuT5")
def load_ru():
    ruT5_model = Rut5BasedTagger()
    return ruT5_model


@st.cache_resource(show_spinner="Loading Bart")
def load_bart():
    bart_model = BartBasedTagger()
    return bart_model


@st.cache_resource(show_spinner="Loading Attention")
def load_attention():
    attention_model = AttentionBasedTagger()
    return attention_model


@st.cache_resource(show_spinner="Loading Clusterization")
def load_clusterization(_emb_model):
    clusterization_model = DBSCANFaissTagger(_emb_model)
    return clusterization_model


def main():
    st.title("🔮 Inference section")

    load_data()

    emb_model = load_ft_emb_model()
    rake_model = load_rake(emb_model)
    ruT5_model = load_ru()
    bart_model = load_bart()
    attention_model = load_attention()
    clusterizer_model = load_clusterization(emb_model)

    input_text = st.text_area(label="Something", label_visibility="hidden", placeholder="Your text is here")
    if st.button("Proceed"):
        if type(input_text) is str:
            with st.expander("RakeBased Tags"):
                with st.spinner("Extracting tags..."):
                    tags = rake_model.extract(input_text, 5)
                st.write(", ".join(tags))

            with st.expander("BartBased Tags"):
                with st.spinner("Extracting tags..."):
                    tags = bart_model.extract(input_text, 5)
                st.write(", ".join(tags))

            with st.expander("ClusterizedBased Tags"):
                with st.spinner("Extracting tags..."):
                    tags = clusterizer_model.extract(input_text, 5)
                st.write(", ".join(tags))

            with st.expander("AttentionBased Tags"):
                with st.spinner("Extracting tags..."):
                    tags = attention_model.extract(input_text, 5)
                st.write(", ".join(tags))

            with st.expander("RuT5Based Tags"):
                with st.spinner("Extracting tags..."):
                    tags = ruT5_model.extract(input_text, 5)
                st.write(", ".join(tags))

            st.balloons()

        else:
            st.error("You need to input a text in the field above")


if __name__ == "__main__":
    main()
