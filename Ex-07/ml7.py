from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

(X, y) = load_breast_cancer(return_X_y=True, as_frame=True)

data = load_breast_cancer(as_frame=True)
X

X["target"] = y
X.corr()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test.keys()

y_train.unique()

cls = GaussianNB()
cls.fit(X_train, y_train)

y_pred = cls.predict(X_test)

print("Accuracy : accuracy_score(y_test, y_pred)")

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

cls.fit(X_train_scaled, y_train)
y_pred_scaled = cls.predict(X_test_scaled)
accuracy_score(y_test, y_pred_scaled)

precision_score(y_test, y_pred_scaled)

print(classification_report(y_test, y_pred, target_names=['0', '1']), '\n')

ConfusionMatrixDisplay(confusion_matrix(y_pred, y_test), display_labels=['1', '0']).plot()

confusion_matrix(y_test, y_pred)
