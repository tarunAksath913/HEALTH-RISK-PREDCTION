import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib  # Added for saving scalers
import os

# --- 1. CONFIGURATION ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

print("STEP 1: Loading Diabetes Data...")
# Define the folder and filename
DATA_FOLDER = 'datasets'
FILE_NAME = 'Diabetes_dataset.csv'

# Combine them into a full path
dataset_path = os.path.join(DATA_FOLDER, FILE_NAME)

print("STEP 1: Loading Diabetes Data...")

try:
    df = pd.read_csv(dataset_path)
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Could not find '{FILE_NAME}' in the '{DATA_FOLDER}' directory.")
    print(f"Looked at: {os.path.abspath(dataset_path)}")
    exit()

# --- 2. PREPROCESSING (Updated for your specific indicators) ---
print("STEP 2: Cleaning & Preparing Data...")

# 1. Define the exact columns to keep based on your request
# We keep the target 'Diabetes_binary' + the selected medical/lifestyle indicators.
# Removed: CholCheck, AnyHealthcare, NoDocbcCost, Education, Income
columns_to_keep = [
    'Diabetes_binary', 
    'HighBP', 
    'HighChol', 
    'BMI', 
    'Smoker', 
    'Stroke', 
    'HeartDiseaseorAttack', 
    'PhysActivity', 
    'Fruits', 
    'Veggies', 
    'HvyAlcoholConsump', 
    'GenHlth', 
    'MentHlth', 
    'PhysHlth', 
    'DiffWalk', 
    'Sex', 
    'Age'
]

# Filter the dataset to only these columns
df = df[columns_to_keep].copy()

target_column = 'Diabetes_binary'

# Drop duplicates if any
df = df.drop_duplicates()

# Separate Features (X) and Target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

print(f"✅ Training on {len(X.columns)} features: {list(X.columns)}")

# 2. Encode Text Columns
# Note: Your specific dataset (BRFSS) is mostly numeric already, 
# but we keep this logic in case you have non-numeric data in the future.
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# 3. Normalize (Scale inputs)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Class Imbalance Handling
# Note: Since you are using the '5050split' dataset, it is likely already balanced.
# However, this code dynamically calculates weights just in case.
class_counts = y.value_counts()
# Ensure we have both classes (0 and 1)
count_0 = class_counts.get(0.0, 1) # default to 1 to avoid div by zero
count_1 = class_counts.get(1.0, 1)

weight_for_0 = 1.0
weight_for_1 = count_0 / count_1
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
            nn.Dropout(0.3),
            
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

# Pass weights to loss function
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. TRAINING LOOP ---
print(f"STEP 3: Training on {len(df)} rows...")

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
    
    # Calculate Accuracy
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    
    # Calculate Sensitivity (Recall)
    true_positives = ((predicted == 1) & (y_test_tensor == 1)).sum().item()
    actual_positives = (y_test_tensor == 1).sum().item()
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
    print(f"🕵️  Sensitivity (Caught Cases): {recall * 100:.2f}%")

# --- 7. SAVE ---
torch.save(model.state_dict(), 'models/diabetes_model_pytorch.pt')
joblib.dump(scaler, 'models/diabetes_scaler.pkl')
joblib.dump(label_encoders, 'models/diabetes_encoders.pkl')
print("\n✅ Model and Scalers saved!")