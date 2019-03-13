from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import pandas as pd
import string
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import pickle

folder_data = 'datsets/'

stopwords=stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    #tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

vectorizer = TfidfVectorizer(
    #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords, #We do better when we keep stopwords
    use_idf=True,
    smooth_idf=False,
    norm=None, #Applies l2 norm smoothing
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.501
    )

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


#Construct tfidf matrix and get relevant scores
tfidf = vectorizer.fit_transform(tweets).toarray()
vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
idf_vals = vectorizer.idf_
idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores

#Get POS tags for tweets and save as a string
tweet_tags = []
for t in tweets:
    tokens = basic_tokenize(preprocess(t))
    tags = nltk.pos_tag(tokens)
    tag_list = [x[1] for x in tags]
    #for i in range(0, len(tokens)):
    tag_str = " ".join(tag_list)
    tweet_tags.append(tag_str)
        #print(tokens[i],tag_list[i])

#We can use the TFIDF vectorizer to get a token matrix for the POS tags
pos_vectorizer = TfidfVectorizer(
    #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None, #We do better when we keep stopwords
    use_idf=False,
    smooth_idf=False,
    norm=None, #Applies l2 norm smoothing
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.501,
    )

#Construct POS TF matrix and get vocab dict
pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}

# Now get other features
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *

sentiment_analyzer = VS()


def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    ##SENTIMENT
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet)  # Get text only

    syllables = textstat.syllable_count(words)  # count syllables in words
    num_chars = sum(len(w) for w in words)  # num chars in words
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet)  # Count #, @, and http://
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    # features = pandas.DataFrame(features)
    return features


def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)


other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                        "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", "vader compound", \
                        "num_hashtags", "num_mentions", "num_urls", "is_retweet"]

feats = get_feature_array(tweets)

#Now join them all up
M = np.concatenate([tfidf,pos,feats],axis=1)

# #Finally get a list of variable names
# variables = ['']*len(vocab)
# for k,v in vocab.items():
#     variables[v] = k
#
# pos_variables = ['']*len(pos_vocab)
# for k,v in pos_vocab.items():
#     pos_variables[v] = k
#
# feature_names = variables+pos_variables+other_features_names

X = pd.DataFrame(M)
y = full_df['class']

with open("tweets.pkl", "wb") as file:
    pickle.dump(X, file)

with open("output.pkl", "wb") as file1:
    pickle.dump(y, file1)


with open("tfidf.pkl", "wb") as file2:
    pickle.dump(tfidf, file2)

with open("pos.pkl", "wb") as file3:
    pickle.dump(pos, file3)

with open("feats.pkl", "wb") as file4:
    pickle.dump(feats, file4)

# select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
# X_ = select.fit_transform(X,y)
#
# x_tr, x_te, y_tr, y_te = train_test_split(X_, y)
# model = LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge',multi_class='ovr')
# model.fit(x_tr, y_tr)
# y_preds = model.predict(x_te)
# report = classification_report( y_te, y_preds )
# print("SVM:")
# print(report)
#
# print("RandomForest")
# model = RandomForestClassifier()
# model.fit(x_tr, y_tr)
# y_preds = model.predict(x_te)
# report = classification_report( y_te, y_preds )
# print(report)


