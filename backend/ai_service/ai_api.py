import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import sys  # Needed for safe logging

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

# --- 2. LOAD RESOURCES ---

# Log to stderr so we don't break the JSON response for Node.js
sys.stderr.write("⏳ Loading AI Brains...\n")

current_dir = os.path.dirname(os.path.abspath(__file__))

# TARGET PATH: Go up 2 levels (ai_service -> backend -> ROOT) to find 'models'
MODEL_DIR = os.path.join(current_dir, '..', '..', 'models')
MODEL_DIR = os.path.normpath(MODEL_DIR)

sys.stderr.write(f"📂 Looking for models in: {MODEL_DIR}\n")

if not os.path.exists(MODEL_DIR):
    sys.stderr.write("❌ ERROR: The models directory was not found! Check your folder structure.\n")

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

    sys.stderr.write("✅ AI Models Loaded & Ready!\n")

except Exception as e:
    sys.stderr.write(f"\n❌ LOADING ERROR: {e}\n")


# --- 3. PREDICTION FUNCTIONS ---

def predict_diabetes(data):
    try:
        # 1. Age Category (1-13)
        age = int(data.get('age', 25))
        if age <= 24: age_cat = 1
        elif age <= 29: age_cat = 2
        elif age <= 34: age_cat = 3
        elif age <= 39: age_cat = 4
        elif age <= 44: age_cat = 5
        elif age <= 49: age_cat = 6
        elif age <= 54: age_cat = 7
        elif age <= 59: age_cat = 8
        elif age <= 64: age_cat = 9
        elif age <= 69: age_cat = 10
        elif age <= 74: age_cat = 11
        elif age <= 79: age_cat = 12
        else: age_cat = 13

        # 2. Features
        features = [
            1.0 if data.get('highBP', False) else 0.0,
            1.0 if data.get('highChol', False) else 0.0,
            float(data.get('bmi', 0)),
            1.0 if data.get('smoker', False) else 0.0,
            1.0 if data.get('stroke', False) else 0.0,
            1.0 if data.get('heartDisease', False) else 0.0,
            1.0 if int(data.get('physActivityDays', 0)) > 0 else 0.0, 
            1.0 if data.get('fruitDaily', False) else 0.0,
            1.0 if data.get('veggieFreq') == "Always" else 0.0, 
            1.0 if data.get('alcohol') in ["Frequently", "Always"] else 0.0,
            float(data.get('genHealth', 3)), 
            float(data.get('mentHealthDays', 0)),
            float(data.get('physHealthDays', 0)),
            1.0 if data.get('diffWalk', False) else 0.0,
            1.0 if data.get('gender') == 'Male' else 0.0, 
            float(age_cat) 
        ]
        
        features_scaled = d_scaler.transform(np.array([features]))
        tensor_input = torch.tensor(features_scaled, dtype=torch.float32)
        with torch.no_grad():
            outputs = diabetes_model(tensor_input)
            probs = torch.softmax(outputs, dim=1) 
            risk_score = probs[0][1].item() * 100 
            
        return {"risk_score": round(risk_score, 2)}

    except Exception as e:
        sys.stderr.write(f"Diabetes Pred Error: {e}\n")
        return {"error": str(e)}

def predict_obesity(data):
    try:
        # Gender
        gender = 1 if data.get('gender') == 'Male' else 0 
        # Family History
        family_hist = 1 if data.get('family_history', False) else 0
        # FAVC
        favc = 1 if data.get('highCalFood', False) else 0
        # FCVC
        veg_map = {"Never": 1.0, "Sometimes": 2.0, "Always": 3.0}
        fcvc = veg_map.get(data.get('veggieFreq'), 2.0)
        # NCP
        ncp = float(data.get('mealsPerDay', 3))
        # CAEC
        caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        caec = caec_map.get(data.get('snackFreq'), 1)
        # SMOKE
        smoke = 1 if data.get('smoker', False) else 0
        # CH2O
        ch2o = float(data.get('waterDaily', 2))
        # SCC
        scc = 1 if data.get('monitorCalories', False) else 0
        # FAF
        days = int(data.get('physActivityDays', 0))
        faf = 0 if days == 0 else (1 if days <= 2 else (2 if days <= 4 else 3))
        # TUE
        hours = int(data.get('techUseHours', 0))
        tue = 0 if hours < 3 else (1 if hours < 6 else 2)
        # CALC
        calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        calc = calc_map.get(data.get('alcohol'), 1)
        # MTRANS
        try:
            mtrans = o_encoders['MTRANS'].transform([data.get('transport', 'Public_Transportation')])[0]
        except:
            mtrans = 0 
            
        features = [gender, float(data.get('age', 25)), float(data.get('height', 1.70)), float(data.get('weight', 70)),
                    family_hist, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans]
        
        features_scaled = o_scaler.transform(np.array([features]))
        tensor_input = torch.tensor(features_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = obesity_model(tensor_input)
            _, predicted_idx = torch.max(outputs, 1)
            class_name = o_targets.inverse_transform([predicted_idx.item()])[0]
            
        return {"obesity_level": class_name}

    except Exception as e:
        sys.stderr.write(f"Obesity Pred Error: {e}\n")
        return {"error": str(e)}

if __name__ == '__main__':
    # This block allows you to test the API file directly if needed
    print("This file is meant to be imported, not run directly.")