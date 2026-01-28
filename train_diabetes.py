import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# --- 1. CONFIGURATION ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

print("STEP 1: Loading Diabetes Data...")
try:
    df = pd.read_csv('datasets/diabetes_prediction_dataset.csv')
except FileNotFoundError:
    print("❌ Error: Could not find 'diabetes_prediction_dataset.csv'.")
    exit()

# --- 2. PREPROCESSING (Option B: Lifestyle Only) ---
print("STEP 2: Cleaning & Preparing Data...")

target_column = 'diabetes'

# 1. DROP the medical columns (The "Option B" change)
# We also drop duplicates if any
df = df.drop_duplicates()
columns_to_drop = [target_column, 'HbA1c_level', 'blood_glucose_level']
X = df.drop(columns=columns_to_drop)
y = df[target_column]

print(f"✅ Training on these features: {list(X.columns)}")

# 2. Encode Text Columns
# We map 'gender' and 'smoking_history' to numbers
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# 3. Normalize (Scale inputs to be roughly -1 to 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Class Imbalance Handling (Essential for Diabetes)
# Most people don't have diabetes, so we tell the model to pay 
# more attention to the ones who do.
class_counts = y.value_counts()
weight_for_0 = 1.0
weight_for_1 = class_counts[0] / class_counts[1]
weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32)
print(f"⚖️  Class Weights applied: {weights}")

# 6. Convert to Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# --- 3. DATASET SETUP ---
class HealthDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = HealthDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 4. DEFINE THE MODEL ---
class DiabetesModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DiabetesModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),  # Increased dropout to prevent overfitting on smaller data
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, num_classes) 
        )
    
    def forward(self, x):
        return self.network(x)

input_features = X.shape[1]
num_classes = 2
model = DiabetesModel(input_features, num_classes)

# We pass the weights to the loss function here
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. TRAINING LOOP ---
print(f"STEP 3: Training on {len(df)} rows with {input_features} features...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f}")

# --- 6. EVALUATION ---
print("\nSTEP 4: Testing...")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    
    # Calculate simple accuracy
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    
    # Calculate Sensitivity (Recall) - How many actual diabetics did we catch?
    true_positives = ((predicted == 1) & (y_test_tensor == 1)).sum().item()
    actual_positives = (y_test_tensor == 1).sum().item()
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
    print(f"🕵️  Sensitivity (Caught Cases): {recall * 100:.2f}%")

# --- 7. SAVE ---
torch.save(model.state_dict(), 'models/diabetes_model_pytorch.pt')
# We also save the scaler so we can scale user input exactly the same way
import joblib
joblib.dump(scaler, 'models/diabetes_scaler.pkl')
joblib.dump(label_encoders, 'models/diabetes_encoders.pkl')
print("\n✅ Model and Scalers saved!") 