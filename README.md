# Spam-classification
spam classification using naive bayes
dataset taken from kaggle total rows(5573)
out of it 1 row was removed
text cleaned and vectorized with countvectorizer 
MultinomialNB via sklearn pipeline
stratified y
results:
Train accuracy: 0.9928
Test accuracy:  0.9865

Confusion Matrix:
[[964   2]
 [ 13 136]]

Classification Report:
              precision  recall  f1-score
ham    (0)      0.99      1.00      0.99
spam   (1)      0.99      0.91      0.95

Compared MultinomialNB, SVM, RandomForest, XGBoost, and KNN. MultinomialNB performed best on this dataset with lowest false negatives on the spam class
