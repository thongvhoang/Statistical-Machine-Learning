# Import librabries
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values

# Split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

Y_train_pred = classifier.predict(X_train)
Y_test_pred = classifier.predict(X_test)

# Visualizing confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, Y_train_pred)
plt.title('Confusion matrix on train set')
sns.heatmap(cm,annot=True,fmt='g',cmap='Blues',annot_kws={"size": 40})
plt.show()
cm = confusion_matrix(Y_test, Y_test_pred)
plt.title('Confusion matrix on test set')
sns.heatmap(cm,annot=True,fmt='g',cmap='Blues',annot_kws={"size": 40})
plt.show()

# Save model
joblib.dump(classifier,'model.pkl')

# Calculate accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_train,Y_train_pred)
print('Accuracy on train set:',accuracy)
accuracy = accuracy_score(Y_test,Y_test_pred)
print('Accuracy on test set:',accuracy)