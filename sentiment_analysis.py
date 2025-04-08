#!/usr/bin/env python3

import argparse
import requests
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

def setup_nltk():
    """Download necessary NLTK resources"""
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    print('NLTK resources downloaded successfully.')

def fetch_news_articles(api_key, query="US tariffs", days=7):
    """
    Fetch news articles from NewsAPI
    Args:
        api_key (str): NewsAPI key
        query (str): Search query
        days (int): Number of days to look back
    Returns:
        pd.DataFrame: DataFrame containing news articles
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Format dates for API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    # API endpoint
    url = f'https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'ok':
            articles = data['articles']
            df = pd.DataFrame(articles)
            print(f"Successfully fetched {len(df)} articles")
            return df
        else:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Error fetching articles: {str(e)}")
        return None

def preprocess_text(text):
    """
    Preprocess text for analysis
    Args:
        text (str): Text to preprocess
    Returns:
        list: List of preprocessed tokens
    """
    if not isinstance(text, str):
        return []

    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def analyze_sentiment(df):
    """
    Perform sentiment analysis on the articles
    Args:
        df (pd.DataFrame): DataFrame containing news articles
    Returns:
        pd.DataFrame: DataFrame with sentiment scores
    """
    sia = SentimentIntensityAnalyzer()

    # Add sentiment scores
    df['sentiment_scores'] = df['title'].apply(lambda x: sia.polarity_scores(str(x)))
    df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])

    # Categorize sentiment
    df['sentiment'] = df['compound_score'].apply(lambda x:
        'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral')

    return df

def train_word2vec(df):
    """
    Train Word2Vec model on article titles
    Args:
        df (pd.DataFrame): DataFrame containing news articles
    Returns:
        Word2Vec: Trained Word2Vec model
    """
    # Preprocess all titles
    processed_titles = df['title'].apply(preprocess_text).tolist()

    # Train Word2Vec model
    model = Word2Vec(
        sentences=processed_titles,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )

    return model

def visualize_embeddings(model, output_dir='picture'):
    """
    Visualize word embeddings using t-SNE
    Args:
        model (Word2Vec): Trained Word2Vec model
        output_dir (str): Directory to save visualization
    """
    # Get word vectors
    words = list(model.wv.key_to_index.keys())
    vectors = np.array([model.wv[word] for word in words])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)

    # Create DataFrame for visualization
    df_tsne = pd.DataFrame(vectors_2d, columns=['x', 'y'])
    df_tsne['word'] = words

    # Create visualization
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_tsne, x='x', y='y')

    # Add word labels
    for i, word in enumerate(words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))

    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'word_embeddings.png'))
    plt.close()

def main():
    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(description='Sentiment Analysis of US Tariffs News')
    parser.add_argument('--api-key', help='NewsAPI key (optional if set in .env file)')
    parser.add_argument('--query', default='US tariffs', help='Search query')
    parser.add_argument('--days', type=int, default=7, help='Number of days to look back')
    parser.add_argument('--output-dir', default='output', help='Output directory for results')
    parser.add_argument('--picture-dir', default='picture', help='Directory for saving visualizations')

    args = parser.parse_args()

    # Get API key from command line or .env file
    api_key = args.api_key or os.getenv('NEWS_API_KEY')
    if not api_key:
        print("Error: No API key provided. Please set NEWS_API_KEY in .env file or use --api-key argument")
        return

    # Setup
    setup_nltk()

    # Fetch articles
    df = fetch_news_articles(api_key, args.query, args.days)
    if df is None:
        return

    # Analyze sentiment
    df = analyze_sentiment(df)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'news_articles.csv'), index=False)

    # Train Word2Vec and visualize
    model = train_word2vec(df)
    visualize_embeddings(model, args.picture_dir)

    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total articles analyzed: {len(df)}")
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts())
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Visualization saved to: {args.picture_dir}")

if __name__ == "__main__":
    main()
