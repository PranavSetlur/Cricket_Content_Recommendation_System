#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.metrics.pairwise import cosine_similarity


# Download necessary NLTK resources
nltk.download("stopwords", quiet = True)
nltk.download("wordnet", quiet = True)
nltk.download("punkt", quiet = True)
nltk.download('averaged_perceptron_tagger', quiet = True)

class CricinfoScraper:
    def __init__(self, url, base_url = None):
        self.url = url
        self.base_url = base_url
    
    def get_articles(self, start_page = 1, num_pages = 1):
        articles = []
        for page_num in range(start_page, start_page + num_pages + 1):
            new_url = f"{self.url}?page={page_num}"
            response = requests.get(new_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            page_articles = self.scrape_page(soup)
            articles.extend(page_articles)
        
        return articles
    
    def scrape_page(self, soup):
        articles = []
        titles = []
        urls = []
        summaries = []
        dates = []
        
        # getting title and link
        for article in soup.find_all('h2', class_='ds-text-title-s ds-font-bold ds-text-typo'):
            title = article.text.strip()
            titles.append(title)
        
            link_tag = article.find_parent('a')
            link = link_tag['href'] if link_tag else ""
            if self.base_url:
                link = self.base_url + link
            urls.append(link)
            
        # getting summaries
        for article in soup.find_all('p', class_='ds-text-compact-s ds-text-typo-mid2 ds-mt-1'):
            summary = article.text.strip()
            summaries.append(summary)
            
        # getting publication date
        for article in soup.find_all('div', class_='ds-leading-[0] ds-text-typo-mid3 ds-mt-1'):
            date_text = article.text.strip()
            date = date_text.split('•')[0]
            dates.append(date)
            
        for title, url, summary, date in zip(titles, urls, summaries, dates):
            articles.append({
                'title': title,
                'link': url,
                'summary': summary,
                'date': date
            })

        return articles
    
    def save_articles(self, articles, filename):
        df = pd.DataFrame(articles)
        df.to_csv(filename, index = False)
    
    def load_and_combine_articles(self, filenames):
        dfs = [pd.read_csv(filename) for filename in filenames]
        df = pd.concat(dfs).drop_duplicates().dropna().reset_index(drop = True)
        
        # converting all text to lower case
        df['title'] = df['title'].str.lower()
        df['summary'] = df['summary'].str.lower()
        
        return df


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.english_stopwords = set(nltk.corpus.stopwords.words('english'))
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"'s(\s|$)", r'\1', text)
        text = text.replace("'", "")
        text = text.strip()
        return text
    
    def tokenize(self, cleaned_text):
        tokens = nltk.word_tokenize(cleaned_text)
        new_tokens = []
        for token in tokens:
            split_token = re.split(r'[^0-9a-zA-Z]+', token)
            split_token = [token for token in split_token if token]
            new_tokens.extend(split_token)
        return new_tokens

    def lemmatize(self, tokens, stopwords = {}):
        lemmatized_tokens = []
        for token in tokens:
            tag = nltk.pos_tag([token])[0][1]
            if tag.startswith('J'):
                tag = wordnet.ADJ
            elif tag.startswith('V'):
                tag = wordnet.VERB
            elif tag.startswith('R'):
                tag = wordnet.ADV
            else:
                tag = wordnet.NOUN

            lemmatized = self.lemmatizer.lemmatize(token, pos=tag)
            if (lemmatized not in stopwords) and (len(lemmatized) >= 2):
                lemmatized_tokens.append(lemmatized)
        return lemmatized_tokens
    
    def preprocess_text(self, text, stopwords={}):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        return self.lemmatize(tokens, stopwords)
    
    def join_tokens(self, tokens):
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_columns):
        for column in text_columns:
            df[column] = df[column].apply(lambda x: self.preprocess_text(x, self.english_stopwords))
            df[column] = df[column].apply(self.join_tokens)
        return df



class FeatureExtractor:
    def __init__(self):
        self.dummy = lambda x : x
            
    def count_vectorizer(self, text):
        vectorizer = CountVectorizer(analyzer = str.split, tokenizer = str.split, preprocessor = self.dummy)
        tf_text = vectorizer.fit_transform(text)
        features = vectorizer.get_feature_names_out().tolist()
        return tf_text, features
    
    def tfidf_vectorizer(self, text):
        vectorizer = TfidfVectorizer(analyzer = str.split, tokenizer = str.split, preprocessor = self.dummy)
        tf_text = vectorizer.fit_transform(text)
        features = vectorizer.get_feature_names_out().tolist()
        return tf_text, features
    
    def top_words_by_topic(self, text, n_topics = 10, n_top_words = 20, seed = 42):
        tf_text, features = self.count_vectorizer(text)
        lda = LatentDirichletAllocation(n_components = n_topics, random_state = seed, learning_method = 'online')
        lda.fit(tf_text)
        topics = lda.components_

        top_words = []
        for i in range(n_topics):
            indices = topics[i].argsort()
            top_indices = indices[-n_top_words : ]
            words = [features[j] for j in top_indices]
            top_words.append(words)
        return top_words
    
    def extract_features(self, df):
        tokenized_df = df.copy()

        tf_title, features_title_tf = self.count_vectorizer(tokenized_df['title'])
        tf_summary, features_summary_tf = self.count_vectorizer(tokenized_df['summary'])

        tfidf_title, features_title_tfidf = self.tfidf_vectorizer(tokenized_df['title'])
        tfidf_summary, features_summary_tfidf = self.tfidf_vectorizer(tokenized_df['summary'])

        tfidf_title_dense = tfidf_title.toarray()
        tfidf_summary_dense = tfidf_summary.toarray()

        df_title_tfidf = pd.DataFrame(tfidf_title_dense, columns = features_title_tfidf)
        df_summary_tfidf = pd.DataFrame(tfidf_summary_dense, columns = features_summary_tfidf)

        vectorized_df = pd.concat([df_title_tfidf, df_summary_tfidf], axis = 1)

        vectorized_df['original_title'] = df['title']
        vectorized_df['original_summary'] = df['summary']
        vectorized_df['link'] = df['link']

        np.save('vectorized_df.npy', vectorized_df)

        return vectorized_df


class Recommender:
    def __init__(self, vectorized_df):
        self.vectorized_df = vectorized_df
        self.numerical_df = vectorized_df.drop(['link', 'original_title', 'original_summary'], axis = 1)
        self.cosine_sim = None
        self.top_20_recs = None
        
    def compute_cosine_similarity(self):
        self.cosine_sim = cosine_similarity(self.numerical_df)
        np.save('cosine_sim.npy', self.cosine_sim)
        return self.cosine_sim
    
    def load_cosine_similarity(self, file_path):
        self.cosine_sim = np.load(file_path)
        return self.cosine_sim
    
    def get_recommendations(self, article_id, top_n = 5):
        if self.cosine_sim is None:
            raise ValueError("Cosine similarity matrix not computed or loaded.")
        
        scores = list(enumerate(self.cosine_sim[article_id]))
        scores = sorted(scores, key=lambda x: x[1], reverse = True)
        scores = scores[1 : top_n + 1]
        article_indices = [i[0] for i in scores]
        return article_indices
    
    def cache_top_recommendations(self, top_n = 20):
        if self.cosine_sim is None:
            raise ValueError("Cosine similarity matrix not computed or loaded.")
        
        n = self.cosine_sim.shape[0]
        self.top_20_recs = np.zeros((n, top_n), dtype = int)

        for i in range(n):
            if i % 100 == 0:
                print(f"Processing article {i}")
            self.top_20_recs[i] = self.get_recommendations(i, top_n = top_n)

        np.save('top_20_recommendations.npy', self.top_20_recs)
        return self.top_20_recs



# Running the code
def run():
    url = 'https://www.espncricinfo.com/ci/content/story/news.html'
    base_url = 'https://www.espncricinfo.com'
    
    # Scrape articles
    scraper = CricinfoScraper(url, base_url)
    articles = scraper.get_articles(1, 1000)
    scraper.save_articles(articles, 'cricinfo_articles.csv')
    
    more_articles = scraper.get_articles(1001, 300)
    scraper.save_articles(more_articles, 'cricinfo_articles_2.csv')
    
    articles_3 = scraper.get_articles(1301, 200)
    scraper.save_articles(articles_3, 'cricinfo_articles_3.csv')
    
    filenames = ['cricinfo_articles.csv', 'cricinfo_articles_2.csv', 'cricinfo_articles_3.csv']
    df = scraper.load_and_combine_articles(filenames)
    df.to_csv('articles_full.csv', index=False)
    
    df = pd.read_csv('articles_full.csv')

    # Preprocess articles
    preprocessor = TextPreprocessor()
    text_columns = ['title', 'summary']
    df = preprocessor.preprocess_dataframe(df, text_columns)
    df.to_csv('tokenized_articles.csv', index=False)

    # Extract features
    tokenized_df = pd.read_csv('tokenized_articles.csv')
    extractor = FeatureExtractor()
    vectorized_df = extractor.extract_features(tokenized_df)

    # Compute top words by topic
    top_words_title = extractor.top_words_by_topic(tokenized_df['title'])
    top_words_summary = extractor.top_words_by_topic(tokenized_df['summary'])

    # Save and cache recommendations
    recommender = Recommender(vectorized_df)
    cosine_sim = recommender.compute_cosine_similarity()
    top_20_recs = recommender.cache_top_recommendations()

if __name__ == "__main__":
    run()
