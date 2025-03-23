# Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP
This project analyzes customer sentiment in women's e-commerce clothing reviews using Natural Language Processing (NLP). The goal is to classify customer reviews as positive, negative, or neutral to help businesses understand customer satisfaction, identify areas for improvement, and enhance decision-making processes.


## Project Overview
This project focuses on performing **Sentiment Analysis** on women's e-commerce clothing reviews using **Natural Language Processing (NLP)** techniques. The goal is to classify customer reviews as **positive**, **negative**, or **neutral** to help businesses understand customer satisfaction, identify areas for improvement, and enhance decision-making processes.

The dataset used contains customer reviews, ratings, and feedback counts, which are analyzed to provide actionable insights. The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Sentiment classification using machine learning models

---

## Dataset
The dataset used in this project is sourced from **Kaggle**:  
[Women's E-commerce Clothing Reviews Dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

### Dataset Description
The dataset contains the following columns:
- **Clothing ID**: Unique identifier for each product.
- **Age**: Age of the reviewer.
- **Title**: Title of the review.
- **Review Text**: Text of the review.
- **Rating**: Rating given by the reviewer (1-5).
- **Recommended IND**: Binary indicator (1 = recommended, 0 = not recommended).
- **Positive Feedback Count**: Number of positive feedbacks for the review.
- **Division Name**: Division of the product.
- **Department Name**: Department of the product.
- **Class Name**: Class of the product.

---

## Project Objectives
1. Clean and preprocess the text data to make it suitable for NLP tasks.
2. Perform exploratory data analysis (EDA) to understand the distribution of ratings, recommended items, and feedback counts.
3. Use NLP techniques to classify reviews into positive, negative, or neutral sentiments.
4. Provide insights to improve product quality, customer service, and marketing strategies.
5. Develop and evaluate machine learning models for sentiment prediction.

---

## Technologies Used
- **Python**: Primary programming language.
- **Libraries**:
  - Pandas, NumPy, Matplotlib, Seaborn (for data manipulation and visualization).
  - NLTK, WordNetLemmatizer, Stopwords (for text preprocessing).
  - Scikit-learn (for machine learning models).
  - WordCloud (for text visualization).
- **Jupyter Notebook**: For interactive coding and visualization.

---

## Project Steps

### Data Preprocessing
The dataset was cleaned and preprocessed to prepare it for analysis. This included:
- Handling missing values.
- Removing special characters and stopwords.
- Converting text to lowercase.
- Lemmatization to reduce words to their base forms.

```python
# Example of text preprocessing
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

df['Cleaned_Review'] = df['Review Text'].apply(preprocess_text)
```
---

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis was performed to understand the dataset better. Key visualizations include:

Distribution of Ratings:

[![Distribution of Ratings](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/DistributionofRating.png)](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/DistributionofRating.png)



Distribution of Recommended Items:
[![Distribution of Recommended Items](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/DistributionofRecommendedItems.png)](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/DistributionofRecommendedItems.png)


Distribution of Positive Feedback Count:
[![Distribution of Positive Feedback Count](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/PositiveFeedbackCount.png)](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/PositiveFeedbackCount.png)


Sentiment Distribution:
[![Sentiment Distribution](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/SentimentDistribution.png)](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/SentimentDistribution.png)



Word Cloud
[![Word Cloud](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/wordcloud.png)](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/wordcloud.png)

---

## Sentiment Classification
### Initial Logistic Regression Model
The initial sentiment classification was performed using TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization. A Logistic Regression model was trained on the TF-IDF features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# TF-IDF Vectorization
tfidf = TffidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['Cleaned_Review'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```


### Handling Class Imbalance with Combined Features
To address the class imbalance between positive and negative reviews, we combined TF-IDF features, Sentiment Scores, and Word2Vec features. This approach enriches the feature set and improves model performance.
```python
from scipy.sparse import hstack

# TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['Cleaned_Review'])

# Sentiment scores
X_sentiment = df['Sentiment_Score'].values.reshape(-1, 1)

# Word2Vec features
X_word2vec = df[word2vec_columns].values

# Combine all features
X_combined = hstack([X_tfidf, X_sentiment, X_word2vec])

# Target variable
y = df['Sentiment']

# Train-test split with combined features
X_train_combined, X_test_combined, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Logistic Regression Model with combined features
model_combined = LogisticRegression()
model_combined.fit(X_train_combined, y_train)

# Evaluation
y_pred_combined = model_combined.predict(X_test_combined)
print(classification_report(y_test, y_pred_combined))
```

![Confusion Matrix](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/ModelEvaluation.png)
---

