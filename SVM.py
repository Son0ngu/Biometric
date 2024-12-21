import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Example placeholder data (replace this with your actual loading logic)
# Ensure these features are loaded or computed properly
altered_features = np.random.rand(50, 5)  # Generating random data as a placeholder
real_features = np.random.rand(50, 5)  # Generating random data as a placeholder

# Validation to ensure the features aren't empty
if altered_features.size == 0 or real_features.size == 0:
    raise ValueError("altered_features or real_features is empty. Check your data source!")

# Creating the input features (X) and labels (y)
X = np.vstack([altered_features, real_features])  # Combine both datasets
y = np.array(['Altered'] * len(altered_features) + ['Real'] * len(real_features))

# Validation to ensure the dataset is not empty
if X.size == 0 or y.size == 0:
    raise ValueError("Combined dataset (X or y) is empty! Cannot proceed with training.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Output results (for testing purposes)
print("SVM training completed.")
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")