
print("======================== Rain Predictor =======================")

# step 1: sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# step 2: log loss
def log_loss(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    y_pred = np.clip(y_pred, 1e-10, 1-1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Step 3: Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib

# step 4: Load data
print("=="*30)
try:
    df=pd.read_csv("Rain_prediction.csv")
    print("Data loaded suceesfully")
    print("Original data :\n", df.head())
except FileNotFoundError:
    print("File not found checl again!")
    exit()
except Exception as e:
    print("Error while loading data",e)
    exit()
print("=="*30)

# STEP 5: INITIAL VISUALIZATION
# INITIAL VISUALIZATION
print("="*60)
print("INITIAL VISUALIZATION")
print("="*60)

plt.figure(figsize=(15, 10))

# Temperature distribution
plt.subplot(2, 3, 1)
plt.hist(df["Temperature_C"].dropna(), bins=10, color='skyblue', edgecolor='black')
plt.title("Temperature Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")

# Humidity distribution
plt.subplot(2, 3, 2)
plt.hist(df["Humidity_Percent"].dropna(), bins=10, color='lightgreen', edgecolor='black')
plt.title("Humidity Distribution")
plt.xlabel("Humidity (%)")
plt.ylabel("Frequency")

# Wind Speed distribution
plt.subplot(2, 3, 3)
plt.hist(df["Wind_Speed_kmh"].dropna(), bins=10, color='salmon', edgecolor='black')
plt.title("Wind Speed Distribution")
plt.xlabel("Wind Speed (km/h)")
plt.ylabel("Frequency")

# Rain_Tomorrow class distribution
plt.subplot(2, 3, 4)
rain_counts = df["Rain_Tomorrow"].value_counts()
plt.bar(rain_counts.index, rain_counts.values, color=['lightblue', 'lightcoral'])
plt.title("Rain Tomorrow - Class Distribution")
plt.xlabel("Rain Tomorrow")
plt.ylabel("Count")

# Correlation heatmap
plt.subplot(2, 3, 5)
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    im = plt.imshow(corr, cmap='coolwarm')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlations")
    plt.colorbar(im)

# Box plot for outliers
plt.subplot(2, 3, 6)
df[numeric_cols].boxplot()
plt.title("Outlier Detection")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

plt.tight_layout()
# step 6: EDA
print("Basic statistic:\n", df.describe())
print("Missing values:\n", df.isnull().sum())
print("columns:", df.columns.tolist())
print("Basic info:", df.info())
print("=="*30)

# step 7: Filling missing values
df["Temperature_C"] = df["Temperature_C"].fillna(df["Temperature_C"].mean())
print("After filling missing values :\n", df.head())
print("=="*30)

# step 8: Conversion of high and low into numeric in humidity
df["Humidity_Percent"] = df["Humidity_Percent"].replace("high", 85)
df["Humidity_Percent"] = df["Humidity_Percent"].replace("low", 45)
df["Humidity_Percent"] = pd.to_numeric(df["Humidity_Percent"], errors="coerce")

# STEP 9: Binning
# Temperature binning
bins_temp = [0, 15, 25, 35, 50]
labels_temp = ["Very Cold", "Cold", "Warm", "Hot"]
df["Temp_Bin"] = pd.cut(df["Temperature_C"], bins=bins_temp, labels=labels_temp)

# Humidity binning
bins_humidity = [0, 40, 60, 80, 100]
labels_humidity = ["Low", "Moderate", "High", "Very High"]
df["Humidity_Bin"] = pd.cut(df["Humidity_Percent"], bins=bins_humidity, labels=labels_humidity)

# Wind Speed binning
bins_wind = [0, 10, 20, 30, 100]
labels_wind = ["Calm", "Breeze", "Windy", "Stormy"]
df["Wind_Bin"] = pd.cut(df["Wind_Speed_kmh"], bins=bins_wind, labels=labels_wind)

print("Binning completed!")
print(df[["Temperature_C", "Temp_Bin", "Humidity_Percent", "Humidity_Bin", "Wind_Speed_kmh", "Wind_Bin"]].head())

# STEP 10: One-Hot-Encoder
temp_dummies = pd.get_dummies(df["Temp_Bin"], prefix="temp").astype(int)
humidity_dummies = pd.get_dummies(df["Humidity_Bin"], prefix="humidity").astype(int)
wind_dummies = pd.get_dummies(df["Wind_Bin"], prefix="wind").astype(int)

# Add dummy columns to dataframe
for col in temp_dummies.columns:
    df[col] = temp_dummies[col]
for col in humidity_dummies.columns:
    df[col] = humidity_dummies[col]
for col in wind_dummies.columns:
    df[col] = wind_dummies[col]

print("One-hot encoding completed!")
print("=="*40)

# STEP 11: Prepare features
# Get all dummy column names
temp_cols = [col for col in df.columns if col.startswith('temp_')]
humid_cols = [col for col in df.columns if col.startswith('humidity_')]
wind_cols = [col for col in df.columns if col.startswith('wind_')]

# Create feature matrices 
X1 = df[temp_cols].values.astype(float)
X2 = df[humid_cols].values.astype(float)
X3 = df[wind_cols].values.astype(float)

# Convert target
Y = df["Rain_Tomorrow"].map({'Yes': 1, 'No': 0}).values.flatten()

print(f"Total samples: {len(Y)}")
print(f"Temperature features: {X1.shape[1]}")
print(f"Humidity features: {X2.shape[1]}")
print(f"Wind features: {X3.shape[1]}")

# STEP 12: Train-Test Split (80-20)
np.random.seed(42)
n = len(df)
indices = np.arange(n)
np.random.shuffle(indices)

train_size = int(n * 0.8)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

# Split data
X1_train, X1_test = X1[train_idx], X1[test_idx]
X2_train, X2_test = X2[train_idx], X2[test_idx]
X3_train, X3_test = X3[train_idx], X3[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

print(f"\nTrain set size: {len(Y_train)}")
print(f"Test set size: {len(Y_test)}")
print(f"Train - No Rain: {np.sum(Y_train==0)}, Rain: {np.sum(Y_train==1)}")
print(f"Test - No Rain: {np.sum(Y_test==0)}, Rain: {np.sum(Y_test==1)}")

# STEP 13: Feature scaling (ONLY on training data)
# For X1
X1_min = X1_train.min(axis=0)
X1_max = X1_train.max(axis=0)
X1_range = X1_max - X1_min
X1_range[X1_range == 0] = 1
X1_train_scaled = (X1_train - X1_min) / X1_range
X1_test_scaled = (X1_test - X1_min) / X1_range

# For X2
X2_min = X2_train.min(axis=0)
X2_max = X2_train.max(axis=0)
X2_range = X2_max - X2_min
X2_range[X2_range == 0] = 1
X2_train_scaled = (X2_train - X2_min) / X2_range
X2_test_scaled = (X2_test - X2_min) / X2_range

# For X3
X3_min = X3_train.min(axis=0)
X3_max = X3_train.max(axis=0)
X3_range = X3_max - X3_min
X3_range[X3_range == 0] = 1
X3_train_scaled = (X3_train - X3_min) / X3_range
X3_test_scaled = (X3_test - X3_min) / X3_range

# Store scaling parameters
scaling_params = {
    'X1_min': X1_min, 'X1_max': X1_max,
    'X2_min': X2_min, 'X2_max': X2_max,
    'X3_min': X3_min, 'X3_max': X3_max
}

# STEP 14: Initialize parameters
m1 = np.zeros(X1_train_scaled.shape[1])
m2 = np.zeros(X2_train_scaled.shape[1])
m3 = np.zeros(X3_train_scaled.shape[1])
b = 0
lr = 0.1
epochs = 1000
n_train = len(Y_train)
loss_history = []

# STEP 15: Training loop
print("\nStarting training...")
for epoch in range(epochs):
    Z = np.dot(X1_train_scaled, m1) + np.dot(X2_train_scaled, m2) + np.dot(X3_train_scaled, m3) + b
    y_pred = sigmoid(Z)
    loss = log_loss(y_pred, Y_train)
    loss_history.append(loss)
    
    # Gradient descent
    dm1 = (1/n_train) * np.dot(X1_train_scaled.T, (y_pred - Y_train))
    dm2 = (1/n_train) * np.dot(X2_train_scaled.T, (y_pred - Y_train))
    dm3 = (1/n_train) * np.dot(X3_train_scaled.T, (y_pred - Y_train))
    db = (1/n_train) * np.sum(y_pred - Y_train)
    
    # Update parameters
    m1 -= lr * dm1
    m2 -= lr * dm2
    m3 -= lr * dm3
    b -= lr * db
    
    if epoch % 200 == 0:
        print(f"Epoch: {epoch}  Loss: {loss:.4f}")

print("\nTraining completed!")
print(f"Final Loss: {loss_history[-1]:.4f}")

# STEP 16: Evaluate on test set
Z_test = np.dot(X1_test_scaled, m1) + np.dot(X2_test_scaled, m2) + np.dot(X3_test_scaled, m3) + b
y_pred_test = sigmoid(Z_test)
y_pred_binary = (y_pred_test >= 0.5).astype(int)

# STEP 17: Confusion Matrix
TP = FP = FN = TN = 0
for actual, predicted in zip(Y_test, y_pred_binary):
    if actual == 1 and predicted == 1:
        TP += 1
    elif actual == 0 and predicted == 0:
        TN += 1
    elif actual == 0 and predicted == 1:
        FP += 1
    elif actual == 1 and predicted == 0:
        FN += 1

# STEP 18: Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*50)
print("MODEL PERFORMANCE ON TEST SET")
print("="*50)
print(f"Confusion Matrix:")
print(f"TP: {TP}  FP: {FP}")
print(f"FN: {FN}  TN: {TN}")
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1_score:.4f}")

# STEP 19: Final Visualizations
plt.figure(figsize=(15, 5))

# Loss curve
plt.subplot(1, 3, 1)
plt.plot(loss_history, color='green', linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True, alpha=0.3)

# Prediction distribution
plt.subplot(1, 3, 2)
plt.hist(y_pred_test[Y_test==0], bins=20, alpha=0.5, label='No Rain', color='blue')
plt.hist(y_pred_test[Y_test==1], bins=20, alpha=0.5, label='Rain', color='red')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Prediction Distribution by Class")
plt.legend()
plt.grid(True, alpha=0.3)

# Bar chart of metrics
plt.subplot(1, 3, 3)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1_score]
colors = ['green', 'blue', 'orange', 'red']
plt.bar(metrics, values, color=colors)
plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# STEP 20: Save model for Streamlit app
model_info = {
    'm1': m1, 'm2': m2, 'm3': m3, 'b': b,
    'temp_cols': temp_cols,
    'humid_cols': humid_cols,
    'wind_cols': wind_cols,
    'scaling_params': scaling_params,
    'binning_info': {
        'temp_bins': bins_temp, 'temp_labels': labels_temp,
        'humidity_bins': bins_humidity, 'humidity_labels': labels_humidity,
        'wind_bins': bins_wind, 'wind_labels': labels_wind
    },
    'performance': {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1_score': f1_score
    }
}

joblib.dump(model_info, "rain_model.pkl")
print("\n" + "="*50)
print("Model saved successfully!")
print("Files created: rain_model.pkl")
print("="*50)

print("\n Model building complete!")
