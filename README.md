# Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP
---

![Project Banner](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/banner.webp) 

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

### Raw Dataset
![Raw Dataset](https://github.com/Uzo-Hill/Sentiment-Analysis-of-Women-s-E-Commerce-Clothing-Reviews-Using-NLP/blob/main/Dataset.PNG) 


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
```python
# Function to categorize sentiment

def categorize_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

# Apply function to create sentiment labels
df['Sentiment'] = df['Rating'].apply(categorize_sentiment)
```


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


## Key Insights:
The sentiment analysis model achieved an accuracy of **82.2%**. Key insights from the analysis include:



The sentiment analysis model achieved the following distribution of predicted sentiments:
- **86.3%** of reviews were classified as **Positive**.
- **7.5%** of reviews were classified as **Negative**.
- **6.2%** of reviews were classified as **Neutral**.

- Positive reviews dominated the dataset, indicating overall customer satisfaction.
- Younger customers (21-30) were more likely to leave positive reviews, while older customers (51-60) were more critical.Potentially indicating different expectations or shopping behaviors across demographics.
- The Dresses department received the most positive feedback, while the Bottoms department had more neutral and negative reviews, suggesting potential areas for improvement.

---

### Conclusion:
This project successfully applied NLP techniques to analyze sentiment in women's e-commerce clothing reviews. The Logistic Regression model achieved an accuracy of 82.2%, with strong performance for positive reviews but room for improvement in classifying neutral and negative reviews. Key insights included the dominance of positive reviews, variations in sentiment across age groups, and differences in sentiment by department.

---

### Recommendations
Based on the analysis, the following recommendations are proposed:

- There's need to improve product quality by addressing issues highlighted in negative reviews, especially in the Bottoms department.

- Leverage positive review vocabulary to reinforce successful product attributes in marketing materials.

- Use insights from age group analysis to tailor marketing strategies for different demographics.

- Implementation of sentiment monitoring by setting up alerts for significant increases in negative sentiment to enable rapid response to emerging issues.

- Further optimize NLP models using deep learning techniques (e.g., LSTMs, BERT) for better contextual understanding.

---

# Thank you for checking out this project! 



