import pandas as pd, string
import numpy as np
import nltk

import re

folder_data = "datsets/"
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stopwords = stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

# stemmer = PorterStemmer()


def cleaning(seq):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    retweet_regex = 'RT'
    punct_regex = '(:|!|")+'
    sp_symbols = '&([a-zA-Z]+|#[0-9]+)[;]?'
    punct = "['|.|,|?|\\-|\"|*|(|)]+"
    parsed_text = re.sub(space_pattern, ' ', seq)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub(retweet_regex, '', parsed_text)
    parsed_text = re.sub(punct_regex, '', parsed_text)
    parsed_text = re.sub(sp_symbols, '', parsed_text)
    parsed_text = re.sub(punct, '', parsed_text)
    return parsed_text.lower()


# def tokenize(tweet):
#     """Removes punctuation & excess whitespace, sets to lowercase,
#     and stems tweets. Returns a list of stemmed tokens."""
#     tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
#     tokens = [stemmer.stem(t) for t in tweet.split()]
#     return tokens


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

dataset = pd.read_csv(folder_data+"labeled_data.csv")
dataset['class'].replace([0], ['hate'], inplace = True)
dataset['class'].replace([1], ['offensive'], inplace = True)
dataset['class'].replace([2], ['none'], inplace = True)
dataset = dataset.drop(['Unnamed: 0','count', 'hate_speech', 'offensive_language', 'neither'], axis=1)

dataset1 = pd.read_csv(folder_data+"wassem_hovy_naacl.csv", sep='\t')
dataset1.rename(columns = {"Label":"class", "Text":"tweet"}, inplace = True)
dataset1 = dataset1.drop(['Tweet_ID','Previous', 'User_ID'], axis=1)

dataset2 = pd.read_csv(folder_data+"train.csv")
dataset2.label.replace([0], ['none'], inplace = True)
dataset2.label.replace([1], ['hate'], inplace = True)
dataset2.rename(columns = {"label":"class"}, inplace = True)
dataset2 = dataset2.drop(['id'], axis=1)

dataset2 = dataset2[dataset2['class']!='none']

dataset3 = pd.read_csv(folder_data+"racism.csv", sep='\t')
a = np.full(dataset3.shape[0], 'racism')
dataset3['class'] = a
dataset3.rename(columns = {dataset3.columns[0]:"tweet"}, inplace = True)

full_df = pd.concat([dataset, dataset1, dataset2, dataset3])
print(full_df.head())
# full_df['class'].replace(['racism', 'sexism'],['hate', 'hate'], inplace = True )

tweets = full_df['tweet']
# Get POS tags for tweets and save as a string
tweet_tags = []
for t in tweets:
    tokens = cleaning(t)

    tweet_tags.append(tokens)

full_df['tweet'] = tweet_tags
print(full_df.head())

full_df.to_csv(folder_data+"dataset.csv", index=False)

