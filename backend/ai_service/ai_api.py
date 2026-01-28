import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. MODEL DEFINITIONS ---

class DiabetesModel(nn.Module):
    def __init__(self, input_size, num_classes=2):
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

# --- 2. LOAD RESOURCES (THE FIX) ---

print("⏳ Loading AI Brains...")

# Get the directory where THIS file (ai_api.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels (backend -> HEALTH RISK PREDICTION) then into 'models'
MODEL_DIR = os.path.join(current_dir, '..', '..', 'models')

# Normalize the path (fixes slash issues on Windows)
MODEL_DIR = os.path.normpath(MODEL_DIR)

print(f"📂 Looking for models in: {MODEL_DIR}")

if not os.path.exists(MODEL_DIR):
    print("❌ ERROR: The models directory was not found!")
    print("   Please check your folder structure.")

try:
    # Load Diabetes Resources
    d_scaler = joblib.load(os.path.join(MODEL_DIR, 'diabetes_scaler.pkl'))
    d_encoders = joblib.load(os.path.join(MODEL_DIR, 'diabetes_encoders.pkl'))
    d_input_size = len(d_scaler.mean_) 
    diabetes_model = DiabetesModel(d_input_size, 2)
    diabetes_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'diabetes_model_pytorch.pt')))
    diabetes_model.eval()

    # Load Obesity Resources
    o_scaler = joblib.load(os.path.join(MODEL_DIR, 'obesity_scaler.pkl'))
    o_encoders = joblib.load(os.path.join(MODEL_DIR, 'obesity_encoders.pkl'))
    o_targets = joblib.load(os.path.join(MODEL_DIR, 'obesity_target_classes.pkl'))
    o_input_size = len(o_scaler.mean_)
    o_classes = len(o_targets.classes_)
    obesity_model = ObesityModel(o_input_size, o_classes)
    obesity_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'obesity_model_pytorch.pt')))
    obesity_model.eval()

    print("✅ AI Models Loaded & Ready!")

except FileNotFoundError as e:
    print(f"\n❌ FILE MISSING: {e}")
    print("   Make sure you ran the training scripts from the ROOT folder!")
except Exception as e:
    print(f"\n❌ LOADING ERROR: {e}")


# --- 3. PREDICTION FUNCTIONS ---

def predict_diabetes(data):
    try:
        df = pd.DataFrame([data])
        
        # Safe Encoding
        if 'gender' in df.columns:
            try:
                df['gender'] = d_encoders['gender'].transform([df['gender'][0]])[0]
            except:
                df['gender'] = 0 # Default fallback
        
        if 'smoking_history' in df.columns:
            try:
                df['smoking_history'] = d_encoders['smoking_history'].transform([df['smoking_history'][0]])[0]
            except:
                df['smoking_history'] = 0 

        features_scaled = d_scaler.transform(df)
        
        tensor_input = torch.tensor(features_scaled, dtype=torch.float32)
        with torch.no_grad():
            outputs = diabetes_model(tensor_input)
            probs = torch.softmax(outputs, dim=1) 
            risk_score = probs[0][1].item() * 100 
            
        return {"risk_score": round(risk_score, 2)}

    except Exception as e:
        print(f"Diabetes Pred Error: {e}")
        return {"error": str(e)}

def predict_obesity(data):
    try:
        df = pd.DataFrame([data])
        
        # Safe Encoding for all text columns
        for col, le in o_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform([df[col][0]])
                except:
                    df[col] = 0
        
        features_scaled = o_scaler.transform(df)
        
        tensor_input = torch.tensor(features_scaled, dtype=torch.float32)
        with torch.no_grad():
            outputs = obesity_model(tensor_input)
            _, predicted_idx = torch.max(outputs, 1)
            
            class_name = o_targets.inverse_transform([predicted_idx.item()])[0]
            
        return {"obesity_level": class_name}

    except Exception as e:
        print(f"Obesity Pred Error: {e}")
        return {"error": str(e)}