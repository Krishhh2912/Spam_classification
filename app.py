import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-trained models
model1 = joblib.load('models\DecisionTreeClassifier().joblib')
model2 = joblib.load('models\LogisticRegression().joblib')
model3 = joblib.load('models\MultinomialNB().joblib')
model4 = joblib.load("models\SVC(kernel='linear').joblib")
cv = joblib.load('models\count_vectorizer.joblib')
# Preprocessing functions
# Create a function to preprocess the input text
def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text)

    # Remove non-alphabetic characters and convert to lowercase
    words = [re.sub('[^A-Za-z]', '', word).lower() for word in words]

    # Initialize a Porter Stemmer
    stemmer = PorterStemmer()

    # Stemming and remove stopwords
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]

    # Join the words back into a single string
    preprocessed_text = ' '.join(words)

    return preprocessed_text

# Classification function
def classify_text(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    preprocessed_feature = cv.transform([preprocessed_text])
    # print(preprocessed_feature)
    preprocessed_feature=preprocessed_feature.toarray()
    # Use the models to make predictions
    prediction1 = model1.predict(preprocessed_feature)
    prediction2 = model2.predict(preprocessed_feature)
    prediction3 = model3.predict(preprocessed_feature)
    prediction4 = model4.predict(preprocessed_feature)

    return prediction1, prediction2, prediction3, prediction4

# Streamlit app
st.markdown("<h1 style='text-align: center; color: red;'>SPAM CHECKER</h1>", unsafe_allow_html=True)

st.subheader('Welcome to spam checker! We help you in determining where message you recieved is SPAM or can be trusted')

# User input
user_input = st.text_area(label='Your message goes here',label_visibility='collapsed',placeholder= 'Enter you message here')
col1, col2, col3 = st.columns(3)
with col2:
    center_button = st.button("Is it SPAM?🤔")
if center_button:
    if user_input:
        # Make predictions
        prediction1, prediction2, prediction3, prediction4 = classify_text(user_input)

        # Display results with labels
        result_labels = ["Not SPAM", "SPAM"]
        results_dict = {
            "Model": ["Multinomial Naive Bayes", "SVC", "Logistic Regression", "Decision Tree Classifier"],
            "Prediction": [
                result_labels[prediction1[0]],
                result_labels[prediction2[0]],
                result_labels[prediction3[0]],
                result_labels[prediction4[0]]
            ],
            
        }

        # Create a DataFrame from the results dictionary
        results_df = pd.DataFrame(results_dict)
        # Display the results as a table
        st.table(results_df)
    else:
        st.warning("Please enter a message to classify.")