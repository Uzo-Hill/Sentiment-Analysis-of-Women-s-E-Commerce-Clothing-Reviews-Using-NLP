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
Distribution of Ratings <!-- Replace with actual image -->

Distribution of Recommended Items:
Distribution of Recommended Items <!-- Replace with actual image -->

Distribution of Positive Feedback Count:
Distribution of Positive Feedback Count <!-- Replace with actual image -->

