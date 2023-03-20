import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load your dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a function to build the Keras model
def build_model(input_dim, n_classes):
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create a custom Keras classifier
class KerasNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, epochs=100, batch_size=32, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None

    def fit(self, X, y):
        input_dim = X.shape[1]
        n_classes = len(np.unique(y))
        self.model = self.build_fn(input_dim, n_classes)

        y_onehot = to_categorical(y)
        self.model.fit(X, y_onehot, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must train the model before making predictions.")

        return np.argmax(self.model.predict(X), axis=-1)

    def score(self, X, y):
        if self.model is None:
            raise RuntimeError("You must train the model before making predictions.")

        y_onehot = to_categorical(y)
        _, accuracy = self.model.evaluate(X, y_onehot, verbose=self.verbose)
        return accuracy

# Create the Keras classifier and use it within a Scikit-Learn pipeline
pipeline = make_pipeline(
    StandardScaler(),
    KerasNNClassifier(build_fn=build_model)
)
