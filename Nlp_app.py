import streamlit as st
from nltk.corpus import wordnet
from textblob import TextBlob

import joblib

emotion = joblib.load(open("emotion_classifier_model.pkl", "rb"))
toxic = joblib.load(open("toxic_classifier_model.pkl", "rb"))
ru_toxic = joblib.load(open("RU_toxic_classifier_model.pkl", "rb"))


def predict_emotions(docx):
    results = emotion.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = emotion.predict_proba([docx])
    return results

def predict_toxic(docx):
    results = toxic.predict([docx])
    return results[0]

def get_prediction_proba_toxic(docx):
    results = toxic.predict_proba([docx])
    return results

def predict_toxic_ru(docx):
    results = ru_toxic.predict([docx])
    return results[0]

def get_prediction_proba_toxic_ru(docx):
    results = ru_toxic.predict_proba([docx])
    return results


def sin_text(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for i in syn.lemmas():
            synonyms.append(i.name())
    return ', '.join(map(str, set(synonyms)))

def ant_text(word):
    antonyms = []

    for syn in wordnet.synsets(word):
        for i in syn.lemmas():
            if i.antonyms():
                antonyms.append(i.antonyms()[0].name())

    return ', '.join(map(str, set(antonyms)))


def main():
    st.title("NLP Project App")

    menu = ["Emotion classifier", "Toxic classifier", "RU Toxic classifier" , 'Synonymizer', 'Antonymizer','Translator',"About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Emotion classifier":
        st.subheader("Emotion in text")

        with st.form(key='Emotion'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            #probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original text")
                st.write(raw_text)

                st.success("Prediction")
                st.write(prediction)



    elif choice == "Toxic classifier":
        st.subheader("Toxic detector")
        with st.form(key='Toxic'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_toxic(raw_text)
            probability = get_prediction_proba_toxic(raw_text)

            with col1:
                st.success("Original text")
                st.write(raw_text)

                if prediction == 1:
                    st.error("Prediction")
                    st.write('Toxic')
                else:
                    st.success("Prediction")
                    st.write('Neutral')

                #st.success("Prediction")
                #st.write(prediction)


            with col2:
                st.success("Prediction probability")
                st.write(probability)
    elif choice == "RU Toxic classifier":
        st.subheader("Russian Toxic detector")
        with st.form(key='Toxic'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_toxic_ru(raw_text)
            probability = get_prediction_proba_toxic_ru(raw_text)

            with col1:
                st.success("Original text")
                st.write(raw_text)


                if prediction == 1:
                    st.error("Prediction")
                    st.write('Toxic')
                else:
                    st.success("Prediction")
                    st.write('Neutral')

            with col2:
                st.success("Prediction probability")
                st.write(probability)

    elif choice == "Synonymizer":
        st.subheader("Synonymizer your word")
        with st.form(key='Synonymizer'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            synom = sin_text(raw_text)
            with col1:
                st.success("Synonyms")
                st.write(synom)


    elif choice == 'Antonymizer':
        st.subheader("Antonymizer your word")
        with st.form(key='Antonymizer'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            ant_t = ant_text(raw_text)
            with col1:
                st.success("Antonym")
                st.write(ant_t)


    elif choice == 'Translator':
        st.subheader("Translator your word")
        with st.form(key='Translator'):
            raw_text = st.text_area("Type here")
            raw_text = TextBlob(raw_text)
            submit_text = st.form_submit_button(label='Submit')
            a = st.selectbox('Select top', ['English', 'Russian', 'France'])

        if submit_text:
            col1, col2 = st.columns(2)

            with col1:
                if a == 'English':
                    tword = raw_text.translate(from_lang='ru', to='en')
                    st.success("Translated")
                    st.write(tword)
                elif a == 'Russian':
                    word = raw_text.translate(from_lang='en', to='ru')
                    st.success("Translated")
                    st.write(word)

                else:
                    word = raw_text.translate(from_lang='ru', to='fr')
                    st.success("Translated")
                    st.write(word)


    else:
        st.subheader("About")
        st.write('Application for nlp')


main()

