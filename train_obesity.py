import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 60

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

print("STEP 1: Loading Obesity Data...")
# Define the folder and filename
DATA_FOLDER = 'datasets'
FILE_NAME = 'Obesity_dataset.csv'

# Combine them into a full path
dataset_path = os.path.join(DATA_FOLDER, FILE_NAME)

print("STEP 1: Loading Obesity Data...")

try:
    df = pd.read_csv(dataset_path)
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Could not find '{FILE_NAME}' in the '{DATA_FOLDER}' directory.")
    print(f"Looked at: {os.path.abspath(dataset_path)}")
    exit()

# --- 2. PREPROCESSING ---
print("STEP 2: Cleaning Data...")

target_column = 'NObeyesdad' 
X = df.drop(columns=[target_column])
y = df[target_column]

# Smart Encoding: We save these encoders to use on the website data later
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Encode Target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tensor Conversion
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# --- 3. DATASET ---
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

# --- 4. MODEL ---
class ObesityModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ObesityModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

input_features = X.shape[1]
num_classes = len(np.unique(y))
model = ObesityModel(input_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. TRAINING ---
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

# --- 6. EVALUATION (New Step) ---
print("\nSTEP 4: Testing...")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    
    # Calculate simple accuracy
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    
    print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

# --- 7. SAVE ---
torch.save(model.state_dict(), 'models/obesity_model_pytorch.pt')
joblib.dump(scaler, 'models/obesity_scaler.pkl')
joblib.dump(label_encoders, 'models/obesity_encoders.pkl')
joblib.dump(target_encoder, 'models/obesity_target_classes.pkl')
print("\n✅ Obesity Model Saved!")