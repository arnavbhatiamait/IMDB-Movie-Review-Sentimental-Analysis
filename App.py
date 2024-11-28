import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# ! Word Index
# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('Simple_rnn_imdb.keras')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

st.title("IMDB Movie Review Sentimental Analysis")
st.write("Enter a movie review to classify it as positive or negative.")
review = st.text_area("Movie Review")
if st.button("Classify"):
    review = preprocess_text(review)
    prediction = model.predict(review)
    st.write(f"Prediction Score {prediction[0][0]}") 
    if prediction > 0.5:
        sentiment="Possitive"
        st.write(f"Semtiment {sentiment}")
        st.write("The review is **Positive**.")
    else:
        sentiment="Negative"
        st.write(f"Semtiment {sentiment}")

        st.write("The review is **Negative**.")
else:
    st.write("Please enter a movie Review") 
