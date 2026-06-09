from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

RANDSEED = 42
#Load Dataset
digits = load_digits()
X = digits.data
y = digits.target
#Split Train - Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDSEED)
#Pipeline Preprocessing MinMax Scaling digabung dengan MLPClassifier
model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(64,64),
        max_iter=500,
        random_state=RANDSEED,
        verbose=True
    ))
])
#Train Model
model.fit(X_train, y_train)
#Test Model
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
#Simpan model ke file
joblib.dump(model, "mymnist.pkl")








