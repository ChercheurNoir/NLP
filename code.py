import streamlit as st
from tensorflow import keras
from keras.preprocessing.text import Tokenizer as KerasTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from bs4 import BeautifulSoup
from requests import get
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')

# Charger les modèles
model_french = keras.models.load_model("best_model_french.h5")
model_english = keras.models.load_model("best_model.h5")

# Chargement des Tokenizers avec spécification de la langue
tokenizer_french = KerasTokenizer(language='french')
tokenizer_english = KerasTokenizer(language='english')

# Définir max_sequence_length en fonction de la langue
max_sequence_length = 41 if tokenizer_french else 20

# Fonction de prédiction
def predict_sentiment(model, tokenizer, text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequences)
    return prediction[0][0]

# Fonction pour extraire le texte d'une page Wikipédia
def extract_text_from_wikipedia(url):
    resp = get(url)
    article_soup = BeautifulSoup(resp.text, 'html.parser')
    paragraphs = article_soup.find_all('p')
    article_text = " ".join(p.text for p in paragraphs)
    return article_text

# Fonction pour effectuer le résumé avec différents algorithmes en anglais
def summarize_text_english(text, summarizer, num_sentences):
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = summarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Fonction pour effectuer le résumé avec différents algorithmes en français
def summarize_text_french(text, summarizer, num_sentences):
    parser = PlaintextParser.from_string(text, SumyTokenizer("french"))
    summarizer = summarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Définir l'image de fond d'écran
background_image = "https://rire.ctreq.qc.ca/wp-content/uploads/sites/2/2018/11/shutterstock_133217741-scaled.jpg"

# Créer des onglets
tabs = ["Sentiment Analysis", "Text Summarization"]
selected_tab = st.radio("Select Task", tabs)

# Ajouter une image de fond d'écran avec la balise HTML
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background: url("{background_image}") no-repeat center center fixed;
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Onglet "Sentiment Analysis"
if selected_tab == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    # Zone de texte pour saisir le texte
    text_input = st.text_area("Enter your text here:")

    # Boutons pour choisir la langue
    language = st.radio("Choose Language:", ['French', 'English'])

    # Prédiction en fonction de la langue choisie
    if st.button("Analyze Sentiment"):
        if language == 'French':
            prediction = predict_sentiment(model_french, tokenizer_french, text_input)
        else:
            prediction = predict_sentiment(model_english, tokenizer_english, text_input)

        # Afficher la prédiction
        st.write("Predicted Sentiment:", "Positive" if prediction > 0.5 else "Negative")
        st.write("Confidence:", round(np.max(prediction) * 100, 2), "%")

# Onglet "Text Summarization"
elif selected_tab == "Text Summarization":
    st.subheader("Text Summarization")

    # Ajouter une zone de texte pour saisir le texte à résumer
    user_text = st.text_area("Enter the text to summarize:")

    # Sélectionnez le modèle de résumé et le pourcentage du résumé
    summarizer_option = st.selectbox("Select Summarizer", ["TextRank", "LexRank", "LSA"])
    percentage_option = st.slider("Select Percentage of Summary", 1, 100, 30, 1)

    # Sélectionnez la langue pour le résumé
    summary_language = st.radio("Select Language for Summary:", ['French', 'English'])

    # Sélectionnez le texte à résumer
    url = st.text_input("Enter Wikipedia URL:")
    if url:
        # Extraire le texte de la page Wikipédia
        text = extract_text_from_wikipedia(url) if not user_text else user_text

        # Définir le nombre de phrases pour le résumé en fonction du pourcentage spécifié
        num_sentences = int(len(nltk.sent_tokenize(text)) * percentage_option / 100)

        # Effectuer le résumé en fonction de l'option choisie et de la langue
        if summary_language == 'French':
            if summarizer_option == "TextRank":
                summary = summarize_text_french(text, TextRankSummarizer, num_sentences)
            elif summarizer_option == "LexRank":
                summary = summarize_text_french(text, LexRankSummarizer, num_sentences)
            elif summarizer_option == "LSA":
                summary = summarize_text_french(text, LsaSummarizer, num_sentences)
        else:
            if summarizer_option == "TextRank":
                summary = summarize_text_english(text, TextRankSummarizer, num_sentences)
            elif summarizer_option == "LexRank":
                summary = summarize_text_english(text, LexRankSummarizer, num_sentences)
            elif summarizer_option == "LSA":
                summary = summarize_text_english(text, LsaSummarizer, num_sentences)

        # Afficher le résumé
        st.subheader("Summary:")
        st.write(summary)



