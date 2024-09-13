from IPython.display import Image
import pandas as pd
import numpy as np
from wordcloud import STOPWORDS
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")
from textblob import TextBlob 
import nltk
import re
from nltk.stem import SnowballStemmer  
from nltk.corpus import stopwords
import string

# Download stopwords data
nltk.download('stopwords')

# Load CSV data
df1=pd.read_csv(r"C:\KLU3.1\ML\StressDetection\dreaddit-test.csv")
df3=pd.read_csv(r"C:\KLU3.1\ML\StressDetection\dreaddit-train.csv")

# Display shapes of DataFrames
print(df1.shape)
print(df3.shape)

# Display random samples from the DataFrames
print(df1.sample())
print(df3.sample())

# Get column names and info for df1
print(df1.columns)
print(df1.info())

# Check for missing values in both DataFrames
print(df1.isnull().sum())
print(df3.isnull().sum())

# Define a function to detect sentiment
def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Create a DataFrame df2 with the "text" column from df1
df2 = df1[["text"]]

# Add a "sentiment" column to df2 using the detect_sentiment function
df2["sentiment"] = df2["text"].apply(detect_sentiment)

# Display the value counts of sentiment scores
print(df2.sentiment.value_counts())

# Define a SnowballStemmer
stemmer = SnowballStemmer("english")

# Define a function to clean text data
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords.words("english")]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Apply the clean function to the "text" column of df2
df2["text"] = df2["text"].apply(clean)

# Display the cleaned text data
print(df2["text"])

# Define a function for generating word clouds
def wc(data, bgcolor):
    plt.figure(figsize=(20, 20))
    mask = np.array(Image.open("stress-954814_960_720.png"))
    wc = WordCloud(background_color=bgcolor, stopwords=STOPWORDS, mask=mask)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis("off")

# Generate a word cloud using the cleaned text data
wc(df2.text, 'white')

# Map label values to "No Stress" and "Stress"
df2["label"] = df3["label"].map({0: "No Stress", 1: "Stress"})

# Select relevant columns in df2
df2 = df2[["text", "label"]]

# Add sentiment scores to df2
df2["sentiment"] = df2["text"].apply(detect_sentiment)

# Display the first few rows of df2
print(df2.head())

# Import seaborn for data visualization
import seaborn as sns

import sklearn
print(sklearn.__version__)




# Create a count plot of the "label" column in df2
sns.countplot(x=df2.label)

# Split the data into features (x) and labels (y)
x = df2.text
y = df2.label

# Import CountVectorizer and other necessary libraries for text vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create a CountVectorizer
vect = CountVectorizer(stop_words="english")

# Transform the text data into numerical features
x = vect.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Create and train a Multinomial Naive Bayes classifier
mb = MultinomialNB()
tahmin = mb.fit(x_train, y_train).predict(x_test)

# Calculate and print the accuracy score of the model
print("Accuracy Score (Multinomial Naive Bayes):", accuracy_score(tahmin, y_test))

# Import DecisionTreeClassifier for another classification model
from sklearn.tree import DecisionTreeClassifier

# Create and train a Decision Tree classifier
d = DecisionTreeClassifier()
d.fit(x_train, y_train)

# Make predictions using the Decision Tree model
tahmin1 = d.predict(x_test)

# Calculate and print the accuracy score of the Decision Tree model
print("Accuracy Score (Decision Tree):", accuracy_score(y_test, tahmin1))

# Define a user input text for prediction
user = "It cleared up and I was okay but. On Monday I was thinking about humans and how the brain works and it tripped me out I got worried that because I was thinking about how the brain works that I would lose sleep and I did. That night was bad just like last time. Also yesterday my sleep was bad I woke up like every hour of the night just like last time. I got kind of scared like I did last time but this time I think that this is fake life which is absurd but I just think about it then get really scared then I think rationally then calm down."

# Transform the user input text into a numerical feature vector
user_vector = vect.transform([user]).toarray()

# Use the Decision Tree model to predict the label for the user input text
output = d.predict(user_vector)

# Print the predicted label
print(output)
