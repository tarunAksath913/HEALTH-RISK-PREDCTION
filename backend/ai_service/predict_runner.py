import sys
import json
from ai_api import predict_diabetes, predict_obesity

# 1. Read data passed from Node.js (as a command line argument)
try:
    input_json = sys.argv[1]
    user_data = json.loads(input_json)
except Exception as e:
    print(json.dumps({"error": "Invalid JSON input", "details": str(e)}))
    sys.exit(1)

# 2. Extract Data for Diabetes Model (Option B: Lifestyle Only)
diabetes_input = {
    'gender': user_data.get('gender', 'Male'),
    'age': float(user_data.get('age', 30)),
    'hypertension': int(user_data.get('hypertension', 0)),
    'heart_disease': int(user_data.get('heart_disease', 0)),
    'smoking_history': user_data.get('smoking_history', 'never'),
    'bmi': float(user_data.get('bmi', 25.0))
}

# 3. Extract Data for Obesity Model
# Note: Mapping Frontend "Yes/No" to 1/0 for specific columns
obesity_input = {
    'Gender': user_data.get('gender', 'Male'),
    'Age': float(user_data.get('age', 30)),
    'Height': float(user_data.get('height', 1.70)),
    'Weight': float(user_data.get('weight', 70)),
    'family_history_with_overweight': 1 if user_data.get('historyObesity') else 0,
    'FAVC': 1 if user_data.get('highCalFood') else 0,  # Freq High Calorie Food
    'FCVC': float(user_data.get('veggieFreq', 2.0)),   # Veggie consumption (1-3)
    'NCP': float(user_data.get('mealsPerDay', 3.0)),   # Main meals (1-4)
    'CAEC': user_data.get('snackFreq', 'Sometimes'),   # Consumption of food between meals
    'SMOKE': 1 if user_data.get('smoking') == 'Yes' else 0,
    'CH2O': float(user_data.get('waterIntake', 2.0)),  # Water (1-3L)
    'SCC': 0, # Calories monitoring (Assumed no for general users)
    'FAF': float(user_data.get('activityFreq', 1.0)),  # Physical Activity Freq (0-3)
    'TUE': float(user_data.get('techUse', 1.0)),       # Time using tech (0-2)
    'CALC': user_data.get('alcohol', 'Sometimes'),
    'MTRANS': user_data.get('transport', 'Public_Transportation')
}

# 4. Run Predictions
diabetes_result = predict_diabetes(diabetes_input)
obesity_result = predict_obesity(obesity_input)

# 5. Combine and Print JSON (This is what Node.js reads)
final_response = {
    "diabetes": diabetes_result,
    "obesity": obesity_result
}

print(json.dumps(final_response))