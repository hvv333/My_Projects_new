from flask import Flask, render_template, request
import pandas as pd
import pickle
import time
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import plotly.express as px

nltk.download('stopwords')  # Download NLTK stopwords dataset

app = Flask(__name__)

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        stemmer = PorterStemmer()
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        return review
    else:
        return ''

# Load model
model = pickle.load(open('models\model_rf.pkl', 'rb'))

# Load the Count Vectorizer
cv = pickle.load(open('models\countVectorizer.pkl', 'rb'))

# Function to predict sentiment
def predict_sentiment(review):
    # Preprocess the input
    processed_review = preprocess_text(review)
    
    if processed_review != '':
        # Transform the preprocessed text into numerical features using CountVectorizer
        review_features = cv.transform([processed_review])
        
        # Predict using the model
        prediction = model.predict(review_features)
        
        # Convert prediction to 1 for positive and 0 for negative
        prediction_label = "Positive" if prediction[0] == 1 else "Negative"
        
        return prediction_label
    else:
        return 'N/A'

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        Review = request.form['review']
        start = time.time()
        prediction_label = predict_sentiment(Review)
        end = time.time()
        prediction_time = round(end - start, 2)
        return render_template('result.html', prediction=prediction_label, time=prediction_time)
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            df = pd.read_csv(uploaded_file)
            df['Predicted Sentiment'] = df['verified_reviews'].apply(predict_sentiment)
            sentiment_distribution = df['Predicted Sentiment'].value_counts().to_dict()

            # Create a pie chart
            fig = px.pie(values=list(sentiment_distribution.values()), names=list(sentiment_distribution.keys()))

            # Convert the plot to HTML
            pie_chart = fig.to_html(full_html=False, default_height=400, default_width=500)

            return render_template('upload_result.html', table=df.to_html(classes='table table-striped'), chart=pie_chart)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)


