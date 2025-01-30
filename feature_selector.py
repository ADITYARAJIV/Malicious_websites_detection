import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold

class FeatureSelector:
    def __init__(self, data):
        self.data = data

        # Convert 'status' column to binary (0 = legitimate, 1 = phishing)
        if "status" in self.data.columns:
            self.data["status"] = self.data["status"].map({"legitimate": 0, "phishing": 1})

    def remove_non_essential_features(self):
        """Remove non-essential features using correlation, mutual information, and variance threshold step by step."""
        
        # Step 1: Remove non-numeric columns
        self.data = self.data.select_dtypes(include=["number"])
        
        # Step 2: Correlation-Based Feature Selection
        correlation_matrix = self.data.corr()
        correlation_with_target = correlation_matrix["status"].abs()

        # Remove weakly correlated features (|correlation| < 0.05)
        weak_features = correlation_with_target[correlation_with_target < 0.05].index.tolist()
        self.data = self.data.drop(columns=weak_features)
        print(f"Removed weak features with correlation < 0.05: {weak_features}")

        # Remove highly correlated features (> 0.9 correlation)
        correlation_threshold = 0.9
        selected_features = correlation_with_target[correlation_with_target > 0.2].index.tolist()

        high_correlation_pairs = set()
        for i in range(len(selected_features)):
            for j in range(i + 1, len(selected_features)):
                feature1, feature2 = selected_features[i], selected_features[j]

                if abs(correlation_matrix.loc[feature1, feature2]) > correlation_threshold:
                    high_correlation_pairs.add((feature1, feature2))

        features_to_remove = set()
        for feature1, feature2 in high_correlation_pairs:
            if correlation_with_target[feature1] > correlation_with_target[feature2]:
                features_to_remove.add(feature2)
            else:
                features_to_remove.add(feature1)

        self.data = self.data.drop(columns=list(features_to_remove))
        print(f"Removed highly correlated features (> 0.9 correlation): {list(features_to_remove)}")

        # Store features after correlation step
        correlation_selected_features = self.data.drop(columns=['status']).columns.tolist()

        # Step 3: Mutual Information Selection (Keep Top 20 Features)
        X = self.data[correlation_selected_features]
        y = self.data["status"]

        mi_scores = mutual_info_classif(X, y, discrete_features='auto')
        mi_df = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})
        mi_df = mi_df.sort_values(by="MI_Score", ascending=False)

        top_n = 15  # Adjust as needed
        top_features = mi_df.head(top_n)["Feature"].tolist()
        self.data = self.data[top_features + ['status']]
        print(f"Selected top {top_n} features based on Mutual Information: {top_features}")

        # Store features after MI selection
        mi_selected_features = self.data.drop(columns=['status']).columns.tolist()

        # Step 4: Variance Threshold Selection
        X = self.data[mi_selected_features]
        selector = VarianceThreshold(threshold=0.03)  # Remove low-variance features
        X_new = selector.fit_transform(X)

        selected_features = X.columns[selector.get_support()].tolist()
        self.data = self.data[selected_features + ['status']]
        print(f"Remaining features after Variance Threshold: {selected_features}")

        # Save final selected features to pickle
        final_selected_features = self.data.drop(columns=['status']).columns.tolist()
        with open('selected_features.pkl', 'wb') as file:
            pickle.dump(final_selected_features, file)

        print(f"Final selected features saved to 'selected_features.pkl'")

        return self.data  # Return final dataset with selected features

if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('dataset_phishing.csv')

    # Initialize FeatureSelector and remove non-essential features
    feature_selector = FeatureSelector(data)
    final_data = feature_selector.remove_non_essential_features()

    print(f"Final selected columns: {final_data.columns.tolist()}")
