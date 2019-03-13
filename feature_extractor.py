import pandas as pd
import gensim
import pickle
import base_algorithms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from  nltk.stem.snowball import EnglishStemmer
directory = 'datsets/'

full_df = pd.read_csv(directory+"dataset.csv")
full_df['class'].replace(['racism', 'sexism'],['hate', 'hate'], inplace = True)


# offensive-none classification
off_none = full_df[full_df['class']!='hate']


def to_string(text):
    if type(text) == float:
        text = str(text)
    return text
off_none['tweet'] = off_none.tweet.apply(lambda row: to_string(row))

SAVED_MODEL = 'models/w2v.pkl'
SAVED_1M_VEC = 'models/w2v_tfidf.pkl'
documents = [tweet.split() for tweet in list(off_none['tweet'])]
model_1M = gensim.models.Word2Vec(documents,
                                      size=300,
                                      window=10,
                                      min_count=1,
                                      workers=4)

model_1M.train(documents, total_examples=len(documents), epochs=10)
with open(SAVED_MODEL, "wb") as f:
        pickle.dump(model_1M, f)

w2v_1M = dict(zip(model_1M.wv.index2word, model_1M.wv.syn0))

my_vectorizer_1M = base_algorithms.TfidfEmbeddingVectorizer(w2v_1M)

my_vectorizer_1M.fit(off_none.tweet)
# with open(SAVED_1M_VEC, "wb") as f:
#     pickle.dump(my_vectorizer_1M, f)

print(len(model_1M.wv.index2word), "vectorized words")

words_list = []
for tweet in list(off_none['tweet']):
    for w in list(tweet.split()):
        words_list.append(w)

words_data = pd.DataFrame(words_list).drop_duplicates()

count = 0
for w in list(words_data[0]):
    if w not in model_1M.wv.index2word:
        count +=1

tw_data = base_algorithms.add_lexical_features(off_none)
lex_cols = ['nbr_characters','nbr_words', 'nbr_ats', 'nbr_hashtags', 'nbr_urls',
            'nbr_letters','nbr_caps', 'nbr_fancy' ]
lex_features = tw_data.as_matrix(columns=lex_cols)

# w2v features, weighted
features_1M_data = my_vectorizer_1M.transform(tw_data.tweet)

# w2v features, no weight
my_vectorizer_1M_noweight = base_algorithms.MeanEmbeddingVectorizer(w2v_1M)
features_1M_not_weighted = my_vectorizer_1M_noweight.transform(tw_data.tweet)



# tfidf features. We impose a feature vector of 90% the size of the dataset,
# so that n_features < n_data. This can be changed

features_tfidf, words_freq, vocab = base_algorithms.get_tfidf_frequencies(tw_data.tweet,
                                                            stem=True,
                                                            remove_stopwords=False,
                                                            ngram_range=(1,3),
                                                          n_features = round((tw_data.shape[0]/10)*9)
                                                         )


# converting the 4K labeled data into one vector,
# in order to keep only the closests to re-run w2v embedding
labels_vector = features_1M_data.mean(axis=0)


# Concatenation
M_1M = np.concatenate([features_1M_data, lex_features],axis=1)
print('w2v features dimension: %s' %str(M_1M.shape))

M_tfidf = np.concatenate([features_tfidf, lex_features],axis=1)
print('Tf-Idf features dimension: %s' %str(M_tfidf.shape))


M_not_weighted = np.concatenate([features_1M_not_weighted, lex_features],axis=1)


SAVED_MODEL_1 = 'models/w2v_selected.pkl'

features_1M_1M = my_vectorizer_1M.transform(off_none.tweet)


# measuring distances between 1M tweets baseline and the average vector
norm_labels = np.linalg.norm(labels_vector)
norm_features = np.linalg.norm(features_1M_1M, axis=1)
dot = labels_vector.dot(features_1M_1M.T)
off_none['similarity'] = dot / (norm_labels * norm_features)
off_none['len'] = [len(t.split()) for t in off_none.tweet]

# designing a subset of the 1M tweets that is closer to the labels vector,
# to re-train w2v embedding
seuil_len = 5
seuil_sim = 0.5
df_selected = off_none.sort_values(by='similarity', ascending=False).drop(off_none[off_none.len < seuil_len].index)
df_selected = df_selected.drop(df_selected[df_selected.similarity < seuil_sim].index)

# generate documents list for training w2v
documents_selected = [tweet.split() for tweet in list(df_selected['tweet'])]

model_selected = gensim.models.Word2Vec(documents_selected,
                                            size=300,
                                            window=10,
                                            min_count=1,
                                            workers=4)
model_selected.train(documents_selected, total_examples=len(documents_selected), epochs=10)

with open(SAVED_MODEL_1, "wb") as f:
        pickle.dump(model_selected, f)


w2v_selected = dict(zip(model_selected.wv.index2word, model_selected.wv.syn0))



SAVED_VEC_1 = 'models/w2v_vector.pkl'


my_vectorizer_selected = base_algorithms.TfidfEmbeddingVectorizer(w2v_selected)

my_vectorizer_selected.fit(df_selected['tweet'], rem_sw=False)

#
# with open(SAVED_VEC_1, "wb") as f:
#     pickle.dump(my_vectorizer_selected, f)


features_selected = my_vectorizer_selected.transform(tw_data.tweet)

M_selected = np.concatenate([features_selected, lex_features], axis=1)

#evaluation

y = tw_data['class']

X_train_1M, X_test_1M, y_train_1M, y_test_1M = train_test_split(M_1M,
                                                                y ,
                                                                random_state=42,
                                                                test_size=0.15)


X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(M_tfidf, y ,
                                                                            random_state=42,
                                                                            test_size=0.15)

X_train_not_weighted, X_test_not_weighted,\
y_train_not_weighted, y_test_not_weighted = train_test_split(M_not_weighted,
                                                     y ,
                                                     random_state=42,
                                                     test_size=0.15)


X_train_selected, X_test_selected,\
y_train_selected, y_test_selected = train_test_split(M_selected,
                                                     y ,
                                                     random_state=42,
                                                     test_size=0.15)


rf_model= RandomForestClassifier(criterion='gini')

rf_model.fit(X_train_tfidf, y_train_tfidf)
y_preds_rf = rf_model.predict(X_test_tfidf)
print("TF-IDF results with Random forest")
print(np.mean(y_preds_rf == y_test_tfidf))

print(classification_report( y_test_tfidf, y_preds_rf ))


rf_model= LinearSVC()

rf_model.fit(X_train_tfidf, y_train_tfidf)
y_preds_rf = rf_model.predict(X_test_tfidf)
print("TF-IDF results with SVM")
print(np.mean(y_preds_rf == y_test_tfidf))

print(classification_report( y_test_tfidf, y_preds_rf ))

print("Neural network results")
model = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(80,40,40,10), activation='relu', random_state=1,learning_rate='adaptive', alpha=1e-6)
model.fit(X_train_tfidf, y_train_tfidf)
y_preds_rf = model.predict(X_test_tfidf)
print(np.mean(y_preds_rf == y_test_tfidf))

print(classification_report( y_test_tfidf, y_preds_rf ))

