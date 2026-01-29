import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. MODEL DEFINITIONS ---
# (Kept exactly as you had them)

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

print("⏳ Loading AI Brains...")

# Get the directory where THIS file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust this path if your models folder is elsewhere. 
# Assuming structure: backend/api/ai_api.py -> models is at backend/models
MODEL_DIR = os.path.join(current_dir, '..', '..', 'models') 
MODEL_DIR = os.path.normpath(MODEL_DIR)

print(f"📂 Looking for models in: {MODEL_DIR}")

if not os.path.exists(MODEL_DIR):
    print("❌ ERROR: The models directory was not found!")

try:
    # Load Diabetes Resources
    d_scaler = joblib.load(os.path.join(MODEL_DIR, 'diabetes_scaler.pkl'))
    # We don't strictly need d_encoders anymore since we manually map, but keeping for safety
    d_encoders = joblib.load(os.path.join(MODEL_DIR, 'diabetes_encoders.pkl'))
    
    # Check input size from scaler (should be 17 for new model)
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

except Exception as e:
    print(f"\n❌ LOADING ERROR: {e}")


# --- 3. PREDICTION FUNCTIONS (UPDATED WITH TRANSLATOR LOGIC) ---

def predict_diabetes(data):
    try:
        # --- TRANSLATOR: React Data -> Diabetes Model Input ---
        
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

        # 2. Extract & Map Other Features
        features = [
            # IMPORTANT: This order MUST match the 'columns_to_keep' list in your training script
            # Diabetes_binary is target, so inputs start at HighBP
            1.0 if data.get('highBP', False) else 0.0,
            1.0 if data.get('highChol', False) else 0.0,
            float(data.get('bmi', 0)),
            1.0 if data.get('smoker', False) else 0.0,
            1.0 if data.get('stroke', False) else 0.0,
            1.0 if data.get('heartDisease', False) else 0.0,
            1.0 if int(data.get('physActivityDays', 0)) > 0 else 0.0, # PhysActivity (Binary)
            1.0 if data.get('fruitDaily', False) else 0.0,
            1.0 if data.get('veggieFreq') == "Always" else 0.0, # Veggies (Binary: 1=Daily)
            1.0 if data.get('alcohol') in ["Frequently", "Always"] else 0.0, # HvyAlcoholConsump
            float(data.get('genHealth', 3)), # 1-5 Scale
            float(data.get('mentHealthDays', 0)),
            float(data.get('physHealthDays', 0)),
            1.0 if data.get('diffWalk', False) else 0.0,
            1.0 if data.get('gender') == 'Male' else 0.0, # Sex (1=Male)
            float(age_cat) # Age (1-13 category)
        ]
        
        # 3. Scale
        # Reshape to (1, 16) or (1, 17) depending on your training
        # If training had 16 inputs (excluding target), this list has 16 items.
        
        features_array = np.array([features])
        features_scaled = d_scaler.transform(features_array)
        
        # 4. Predict
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
        # --- TRANSLATOR: React Data -> Obesity Model Input ---
        
        # 1. Map Categorical Texts to Numbers (using logic or encoders)
        
        # Gender
        gender = 1 if data.get('gender') == 'Male' else 0 
        
        # Family History
        family_hist = 1 if data.get('family_history', False) else 0
        
        # FAVC (Frequent High Calorie)
        favc = 1 if data.get('highCalFood', False) else 0
        
        # FCVC (Veggies Frequency 1-3)
        veg_map = {"Never": 1.0, "Sometimes": 2.0, "Always": 3.0}
        fcvc = veg_map.get(data.get('veggieFreq'), 2.0)
        
        # NCP (Meals) - raw number
        ncp = float(data.get('mealsPerDay', 3))
        
        # CAEC (Snacking)
        # Assuming encoder order: no=0, Sometimes=1, Frequently=2, Always=3
        # Ideally we use: o_encoders['CAEC'].transform(...) but manual is safer for single inputs
        caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        # Fallback to 'Sometimes' (1) if not found
        caec = caec_map.get(data.get('snackFreq'), 1)
        
        # SMOKE
        smoke = 1 if data.get('smoker', False) else 0
        
        # CH2O (Water)
        ch2o = float(data.get('waterDaily', 2))
        
        # SCC (Monitor Calories)
        scc = 1 if data.get('monitorCalories', False) else 0
        
        # FAF (Physical Activity Freq 0-3)
        days = int(data.get('physActivityDays', 0))
        if days == 0: faf = 0
        elif days <= 2: faf = 1
        elif days <= 4: faf = 2
        else: faf = 3
        
        # TUE (Tech Use 0-2)
        hours = int(data.get('techUseHours', 0))
        if hours < 3: tue = 0
        elif hours < 6: tue = 1
        else: tue = 2
        
        # CALC (Alcohol)
        calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        calc = calc_map.get(data.get('alcohol'), 1)
        
        # MTRANS (Transport)
        # Using the actual encoder is best here because the order varies
        try:
            mtrans_label = data.get('transport', 'Public_Transportation')
            # Transform expects a list, returns an array
            mtrans = o_encoders['MTRANS'].transform([mtrans_label])[0]
        except:
            mtrans = 0 # Fallback
            
        # Assemble Feature Vector (Order MUST match training columns)
        # ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
        # 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 
        # 'CALC', 'MTRANS']
        
        features = [
            gender,
            float(data.get('age', 25)),
            float(data.get('height', 1.70)),
            float(data.get('weight', 70)),
            family_hist,
            favc,
            fcvc,
            ncp,
            caec,
            smoke,
            ch2o,
            scc,
            faf,
            tue,
            calc,
            mtrans
        ]
        
        features_array = np.array([features])
        features_scaled = o_scaler.transform(features_array)
        
        tensor_input = torch.tensor(features_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = obesity_model(tensor_input)
            _, predicted_idx = torch.max(outputs, 1)
            class_name = o_targets.inverse_transform([predicted_idx.item()])[0]
            
        return {"obesity_level": class_name}

    except Exception as e:
        print(f"Obesity Pred Error: {e}")
        # Print detailed error for debugging
        import traceback
        traceback.print_exc()
        return {"error": str(e)}