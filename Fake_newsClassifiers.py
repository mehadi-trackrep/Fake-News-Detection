import pandas as pd
df = pd.read_csv('fake_news_dataset.csv')
col = ['URLs','Headline','Body','Label']
df = df[col]
### Avoiding null all body rows --
df = df[pd.notnull(df['Body'])]
# plt.
df


### using a new column called category_id 
df['category_id'] = df['URLs'].factorize()[0]
df['category_id']
category_id_df = df[['URLs', 'category_id']].drop_duplicates().sort_values('category_id')
# category_id_df
category_to_id = dict(category_id_df.values)
# category_to_id
id_to_category = dict(category_id_df[['category_id', 'URLs']].values)
# id_to_category
print(df.head())

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('Label').Body.count().plot.bar(ylim=0)
plt.show()


###Feature extraction begins here
###Tf-idf is a feature extraction process to vectorize the body text and store as array
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
features = tfidf.fit_transform(df.Body).toarray() ### Transforming the Body column texts to tf-idf weighted array
labels = df.category_id
print(features.shape) ### Finally finding the total row x features(term/word)

###At first train with countvector features to feed in MultinomialNB classifier
from sklearn.naive_bayes import MultinomialNB
## Feature methods importing here ..
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
###Here, the body of the training section..
X_train, X_test, y_train, y_test = train_test_split(df['Body'], df['URLs'], random_state = 0)
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)


###Here we used cross_validation for the 4 models ...
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

### Model list:-
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

CV = 5

cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


### Here we used MLP classifier..
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

clf_output = clf.fit(data_train, targets_train)

print(clf.score(data_test, targets_test))