import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = 'tesseract.exe'

# Function to read text from an image
def read_text_from_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Use adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the binarized image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Assume the largest contour is the message text
    largest_contour = contours[0]

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Extract the region of interest (message text)
    roi = gray[y:y + h, x:x + w]

    config = '-c "tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !\'"'

    # Use Pytesseract to extract text from the ROI
    text = pytesseract.image_to_string(roi, config=config)

    return text.strip()

# Load your dataset
data = pd.read_csv('dataset_1.csv')

# Assuming your dataset has columns 'TEXT' and 'Sentiment' (0 for not fraud, 1 for fraud)
X = data['TEXT']
y = data['Sentiment']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using Bag of Words
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

# Function to predict fraud
def predict_fraud(text):
    if not text:
        return -1  # Return -1 for invalid text
    processed_text = vectorizer.transform([text])
    prediction = nb_model.predict(processed_text)
    return prediction[0]

def main():
    # Set app title and layout
    st.set_page_config(page_title="Fraudulent Text Detector", layout="centered")

    # Add custom CSS for app title
    st.markdown(
        """
        <style>
        h1 {
            text-align: center;
            color: #4285F4;
            font-family: Arial, sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add app title
    st.markdown("<h1>Fraudulent Text Detector</h1>", unsafe_allow_html=True)

    # Add custom CSS for file uploader
    st.markdown(
        """
        <style>
        .file-uploader {
            background-color: #F1F1F1;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add upload prompt
    st.markdown(
        "<p style='text-align: center; font-weight: bold;'>Upload the screenshot that you want to verify</p>",
        unsafe_allow_html=True,
    )

    # File uploader with custom CSS
    uploaded_file = st.file_uploader(
        "ㅤ",
        type=["jpg", "png", "jpeg"],
        help="Upload a screenshot to verify its text",
        disabled=False,
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Resize the image to a maximum width of 500 pixels
        max_width = 500
        width, height = image.size
        if width > max_width:
            new_height = int(height * (max_width / width))
            image = image.resize((max_width, new_height))

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Read text from the image
        text = read_text_from_image(image)

        # Display the extracted text with custom CSS
        st.markdown(
            "<h3 style='color: #4285F4;'>Text Extracted From Image:</h3>",
            unsafe_allow_html=True,
        )
        st.write(text)

        # Predict fraud with custom CSS
        prediction = predict_fraud(text)
        st.markdown(
            "<h3 style='color: #4285F4;'>Prediction:</h3>", unsafe_allow_html=True
        )
        if prediction == 0:
            st.write("The text is not fraudulent.")
        else:
            st.write("The text is potentially fraudulent.")

if __name__ == "__main__":
    main()
