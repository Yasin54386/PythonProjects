import os
import tensorflow as tf
import warnings

# Suppress TensorFlow and Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Step 1: Load the Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
column_names = [
    'id', 'diagnosis', 'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
    'mean_smoothness', 'mean_compactness', 'mean_concavity', 'mean_concave_points',
    'mean_symmetry', 'mean_fractal_dimension', 'radius_se', 'texture_se', 'perimeter_se',
    'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
    'symmetry_se', 'fractal_dimension_se', 'worst_radius', 'worst_texture', 'worst_perimeter',
    'worst_area', 'worst_smoothness', 'worst_compactness', 'worst_concavity',
    'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
]

# Load the data
data = pd.read_csv('wdbc.data', header=None, names=column_names)

# Step 2: Data Preprocessing
# Drop 'id' column and map diagnosis to binary (M = 1, B = 0)
data = data.drop('id', axis=1)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for ANN and some ML models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Model Development and Evaluation

# Initialize dictionaries to store results
results = {}

# 3.1 Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
results['Decision Tree'] = {
    'Accuracy': accuracy_score(y_test, dt_predictions),
    'Precision': precision_score(y_test, dt_predictions),
    'Recall': recall_score(y_test, dt_predictions),
    'F1-Score': f1_score(y_test, dt_predictions)
}

# 3.2 Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
results['Random Forest'] = {
    'Accuracy': accuracy_score(y_test, rf_predictions),
    'Precision': precision_score(y_test, rf_predictions),
    'Recall': recall_score(y_test, rf_predictions),
    'F1-Score': f1_score(y_test, rf_predictions)
}

# 3.3 Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
results['Naive Bayes'] = {
    'Accuracy': accuracy_score(y_test, nb_predictions),
    'Precision': precision_score(y_test, nb_predictions),
    'Recall': recall_score(y_test, nb_predictions),
    'F1-Score': f1_score(y_test, nb_predictions)
}

# 3.4 Artificial Neural Network (ANN) using Keras
ann_model = Sequential()
ann_model.add(Dense(units=32, activation='relu', input_dim=X_train.shape[1]))
ann_model.add(Dense(units=16, activation='relu'))
ann_model.add(Dense(units=1, activation='sigmoid'))

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

ann_predictions = (ann_model.predict(X_test) > 0.5).astype(int)

results['ANN'] = {
    'Accuracy': accuracy_score(y_test, ann_predictions),
    'Precision': precision_score(y_test, ann_predictions),
    'Recall': recall_score(y_test, ann_predictions),
    'F1-Score': f1_score(y_test, ann_predictions)
}

# 4. Print Classification Reports
for model, result in results.items():
    print(f"{model} Model Performance:")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Precision: {result['Precision']:.4f}")
    print(f"Recall: {result['Recall']:.4f}")
    print(f"F1-Score: {result['F1-Score']:.4f}")
    print("\n")

# 5. Visualization of Metrics Comparison

# Prepare the results for visualization
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Data for each model
dt_scores = list(results['Decision Tree'].values())
rf_scores = list(results['Random Forest'].values())
nb_scores = list(results['Naive Bayes'].values())
ann_scores = list(results['ANN'].values())

# Define the position for the bars
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

# Plotting the bar chart and saving the figure
fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, dt_scores, width, label='Decision Tree')
rects2 = ax.bar(x - 0.5*width, rf_scores, width, label='Random Forest')
rects3 = ax.bar(x + 0.5*width, nb_scores, width, label='Naive Bayes')
rects4 = ax.bar(x + 1.5*width, ann_scores, width, label='ANN')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.close()

# Plot individual bar charts for each metric
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, metric in enumerate(metrics):
    plt.figure(figsize=(6, 4))
    plt.bar(['Decision Tree', 'Random Forest', 'Naive Bayes', 'ANN'],
            [dt_scores[i], rf_scores[i], nb_scores[i], ann_scores[i]],
            color=['blue', 'green', 'orange', 'red'])
    plt.title(f'Comparison of {metric}')
    plt.ylabel(f'{metric} Score')
    plt.tight_layout()
    # Save each individual plot
    plt.savefig(f'{metric}_comparison.png')
    plt.close()
