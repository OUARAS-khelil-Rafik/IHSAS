# Importation des biblioth�ques n�cessaires
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Chargement des donn�es
test = pd.read_csv("./Dataset/testing_data.csv")
train = pd.read_csv("./Dataset/training_data.csv")

# S�paration des donn�es en texte et �tiquettes
X_train = train['tweet']
y_train = train['sentiment']

X_test = test['tweet']
y_test = test['sentiment']

# Extraction des caract�ristiques avec TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), stop_words=None, norm="l2")
#vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), stop_words=None)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Cr�ation du mod�le Naive Bayes (MultinomialNB)
#model = MultinomialNB()
#model = BernoulliNB(alpha=1.0)
#model = ComplementNB()

# Cr�ation d'autre mod�les
#model=tree.DecisionTreeClassifier()
#model=KNeighborsClassifier(n_neighbors=1)
model = SVC(kernel="linear", C=1.0)
#model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=20, activation = "logistic")

# Entra�nement du mod�le
model.fit(X_train_vec, y_train)

# Pr�dictions sur l'ensemble de test
y_pred = model.predict(X_test_vec)

# �valuation du mod�le
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Affichage des r�sultats
print(f"Pr�cision du mod�le: {accuracy:.2%}")
print("Matrice de confusion:\n", conf_matrix)
print("Rapport de classification:\n", class_report)
