import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, data, target_column, feature_file='selected_features.pkl'):
        """
        Initializes the ModelTrainer class with dataset and target column.
        
        :param data: pandas DataFrame, input dataset
        :param target_column: str, name of the target column in the dataset
        :param feature_file: str, filename of the pickle file containing selected features
        """
        self.data = data
        self.target_column = target_column
        self.feature_file = feature_file
        self.X = None
        self.y = None
        self.selected_features = None
        self.model = None

    def load_selected_features(self):
        """
        Loads the selected features from the pickle file.
        """
        with open(self.feature_file, 'rb') as file:
            self.selected_features = pickle.load(file)
            print(f"Loaded selected features from {self.feature_file}")

    def preprocess_data(self):
        """
        Preprocesses the data by selecting features and target variable.
        """
        # Ensure target column is in binary format (already converted in feature selection step)
        self.X = self.data[self.selected_features] #Use only selected features
        self.y = self.data[self.target_column].map({"legitimate": 0, "phishing": 1})  # Convert target to binary

    def train_model(self):
        """
        Trains the model using RandomForestClassifier on the selected features.
        """
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Scaling the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Predict on test set
        y_pred = self.model.predict(X_test_scaled)

        # Generate classification report
        report = classification_report(y_test, y_pred)
        print("Classification Report:\n", report)

# Example Usage:
if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('dataset_phishing.csv')

    # Initialize the model trainer class
    model_trainer = ModelTrainer(data, 'status')

    # Load the selected features
    model_trainer.load_selected_features()

    # Preprocess the data
    model_trainer.preprocess_data()

    # Train the model
    model_trainer.train_model()

    print("Model training complete!")
