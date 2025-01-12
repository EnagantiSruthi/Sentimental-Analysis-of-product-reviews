import streamlit as st
import pandas as pd
import joblib
import requests
import threading
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import io
import base64

# Load saved models and vectorizer
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/random_forest_model.pkl')  # Replace with your best model

# Preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ''  # Handle NaN values
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
    else:
        text = str(text)  # Convert non-string values (like float) to string
    return text

# Function to analyze reviews from a file
def analyze_file(uploaded_file):
    try:
        # Read file into a Pandas DataFrame
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            data = pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["review"])
        else:
            st.error("Unsupported file format. Please upload a .csv or .txt file.")
            return None, 0, 0, 0, 0

        # Ensure there's a 'review' column in the data
        if 'review' not in data.columns:
            if len(data.columns) == 1:  # Assume the single column contains reviews
                data.columns = ['review']
            else:
                st.error("The file must contain a column named 'review' or a single column of reviews.")
                return None, 0, 0, 0, 0

        # Drop empty reviews
        data.dropna(subset=['review'], inplace=True)
        data = data[data['review'].str.strip().astype(bool)]  # Remove empty or whitespace-only strings

        # Preprocess and analyze reviews
        data['cleaned_review'] = data['review'].apply(preprocess_text)
        X = vectorizer.transform(data['cleaned_review'])
        data['sentiment'] = model.predict(X)

        # Calculate total, positive, negative, and neutral reviews
        total_reviews = len(data)
        positive_reviews = sum(data['sentiment'] == 'positive')
        negative_reviews = sum(data['sentiment'] == 'negative')
        neutral_reviews = sum(data['sentiment'] == 'neutral')

        # Check for discrepancies
        unclassified_reviews = total_reviews - (positive_reviews + negative_reviews + neutral_reviews)
        if unclassified_reviews > 0:
            st.warning(f"Some reviews ({unclassified_reviews}) were not classified as positive, negative, or neutral.")

        # Return results
        return data, total_reviews, positive_reviews, negative_reviews, neutral_reviews

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None, 0, 0, 0, 0




# Global variable to track progress
progress = 0

def convert_to_review_url(base_url):
    parts = base_url.split('/p/')
    if len(parts) != 2:
        raise ValueError("URL format is not as expected.")
    review_page_url = parts[0] + '/product-reviews/' + parts[1]
    review_page_url = review_page_url.split('?')[0] + '?' + '&'.join(
        [param for param in base_url.split('?')[1].split('&') if param.startswith('pid=') or param.startswith('lid=') or param.startswith('marketplace=')])
    return review_page_url

def scrape_and_clean_data(base_url):
    global progress
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Accept-Language': 'en-us,en;q=0.5'
    }

    customer_names = []
    review_title = []
    ratings = []
    comments = []

    review_url = convert_to_review_url(base_url)
    total_pages = 50  # Set to 50 pages to get approximately 500 reviews (10 reviews per page)
    
    for i in range(1, total_pages + 1):
        current_url = f"{review_url}&page={i}"
        page = requests.get(current_url, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')

        names = soup.find_all('p', class_='_2NsDsF AwS1CA')
        for name in names:
            customer_names.append(name.get_text())

        title = soup.find_all('p', class_='z9E0IG')
        for t in title:
            review_title.append(t.get_text())

        rat = soup.find_all('div', class_='XQDdHH Ga3i8K')
        for r in rat:
            rating = r.get_text()
            if rating:
                ratings.append(rating)
            else:
                ratings.append('0')

        cmt = soup.find_all('div', class_='ZmyHeo')
        for c in cmt:
            comment_text = c.div.div.get_text(strip=True)
            comments.append(comment_text)

        progress = int((i / total_pages) * 100)  # Update progress based on total pages processed
        st.session_state.progress_bar.progress(progress)  # Update progress bar in Streamlit

    min_length = min(len(customer_names), len(review_title), len(ratings), len(comments))
    customer_names = customer_names[:min_length]
    review_title = review_title[:min_length]
    ratings = ratings[:min_length]
    comments = comments[:min_length]

    data = {
        'Customer Name': customer_names,
        'Review Title': review_title,
        'Rating': ratings,
        'Comment': comments
    }

    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)

    # Data Cleaning Steps
    def clean_text(text):
        text = re.sub(r"(http\S+|www\S+)", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text

    df['clean_text'] = df['Comment'].apply(clean_text)

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        tokens = tokenizer.tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    df['Cleaned_text'] = df['clean_text'].apply(preprocess_text)

    selected_columns = ['Customer Name', 'Review Title', 'Rating', 'Comment', 'Cleaned_text']
    cleaned_df = df[selected_columns]
    cleaned_df.to_csv('product_reviews.csv', index=False)

    progress = 100  # Ensure progress is set to 100% when done
    st.session_state.progress_bar.progress(100)  # Ensure progress bar reaches 100%

def analyze_sentiment(comment):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(comment)
    return sentiment_scores

def search_comments(csv_file, criteria, num_reviews):
    df = pd.read_csv(csv_file)
    df['Cleaned_text'] = df['Cleaned_text'].fillna('')
    relevant_comments = []
    for _, row in df.iterrows():
        comment = row['Cleaned_text'].lower()
        if any(word in comment for word in criteria):
            relevant_comments.append(row['Comment'])
            if len(relevant_comments) == num_reviews:
                break  # Stop after reaching the specified number of reviews
    return relevant_comments

def analyze_sentiments(comments):
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    for comment in comments:
        sentiment_scores = analyze_sentiment(comment)
        if sentiment_scores['compound'] >= 0.05:
            sentiments['positive'] += 1
        elif sentiment_scores['compound'] <= -0.05:
            sentiments['negative'] += 1
        else:
            sentiments['neutral'] += 1
    return sentiments

def generate_pie_chart(sentiments):
    labels = sentiments.keys()
    sizes = sentiments.values()
    explode = (0.1, 0, 0)  # Explode the 1st slice (positive sentiment)

    plt.figure(figsize=(4, 4))  # Adjust the size of the pie chart
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Save the plot to a string buffer and encode it in base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pie_chart_url = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return pie_chart_url
# Streamlit UI

# Application Title
st.markdown("<h1 style='text-align: center;',>Sentiment Analysis of Product Reviews</h1>", unsafe_allow_html=True)
st.sidebar.title("Sentiment Analysis Options")

# Input options
option = st.sidebar.radio("Choose an option:", ("Analyze Text", "Analyze File", "Analyze URL"))

if option == "Analyze Text":
    user_text = st.text_area("Enter a review:")
    if st.button("Analyze"):
        if user_text:
            cleaned_text = preprocess_text(user_text)
            X = vectorizer.transform([cleaned_text])
            prediction = model.predict(X)[0]

            # Display sentiment with emoji
            if prediction == 'positive':
                st.success(f"Predicted Sentiment: {prediction} 😊")
            elif prediction == 'negative':
                st.success(f"Predicted Sentiment: {prediction} 😞")
            else:
                st.success(f"Predicted Sentiment: {prediction} 😐")
        else:
            st.error("Please enter some text.")


elif option == "Analyze File":
    uploaded_file = st.file_uploader("Upload a file (.csv or .txt):", type=["csv", "txt"])
    st.markdown("Upload a file containing  your dataset, ensuring that the file has a column named 'review' for sentiment analysis.")
    if st.button("Analyze"):
        if uploaded_file:
            # Unpack five returned values from analyze_file
            result, total, positive, negative, neutral = analyze_file(uploaded_file)
            
            if result is not None:
                st.write("Results:")
                st.dataframe(result[['review', 'sentiment']])  # Show original reviews and their sentiments
                st.success(f"Total Reviews: {total}")
                st.success(f"Positive Reviews: {positive}")
                st.success(f"Negative Reviews: {negative}")
                st.success(f"Neutral Reviews: {neutral}")
        else:
            st.error("Please upload a valid file.")
elif option == "Analyze URL":
    url = st.text_input("Enter Product URL", "")

    if st.button("Start Analysis"):
       if url:
          st.session_state.progress_bar = st.progress(0)
          scrape_and_clean_data(url)
          st.success("Data scraping and cleaning completed!")
       else:
          st.error("Please enter a valid URL.")

# Progress Bar
    st.subheader("Scraping Progress")
#st.progress(0)

    criteria = st.text_input("Enter Criteria for Reviews (comma-separated)", "")
    num_reviews = st.number_input("Number of Reviews", min_value=1, max_value=100, value=10)

    if st.button("Perform Analysis"):
       if criteria:
           criteria_list = [c.strip() for c in criteria.split(',')]
           relevant_comments = search_comments('product_reviews.csv', criteria_list, num_reviews)

           if not relevant_comments:
               st.write(f"No reviews found for criteria: {criteria}")
           else:
               sentiments = analyze_sentiments(relevant_comments)
               pie_chart_url = generate_pie_chart(sentiments)

               overall_sentiment = "Worth Buying" if sentiments['positive'] > (sentiments['negative'] + sentiments['neutral']) else "Not Worth Buying"

               st.write(f"Overall Sentiment: {overall_sentiment}")
               st.write(f"Positive: {sentiments['positive']}, Negative: {sentiments['negative']}, Neutral: {sentiments['neutral']}")
               st.image("data:image/png;base64," + pie_chart_url)
               st.write("Relevant Reviews:")
               for comment in relevant_comments:
                  st.write(f"- {comment}")
    else:
         st.error("Please enter criteria for the analysis.")