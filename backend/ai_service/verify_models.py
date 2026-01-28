from ai_api import predict_diabetes, predict_obesity

print("--- 🧪 TESTING DIABETES MODEL (Lifestyle Only) ---")
# 1. Test Case: High Risk Person
high_risk_person = {
    'gender': 'Male',
    'age': 55,
    'hypertension': 1,
    'heart_disease': 0,
    'smoking_history': 'current',
    'bmi': 32.5
}

# 2. Test Case: Low Risk Person
low_risk_person = {
    'gender': 'Female',
    'age': 22,
    'hypertension': 0,
    'heart_disease': 0,
    'smoking_history': 'never',
    'bmi': 21.0
}

print(f"Testing High Risk Input: {high_risk_person}")
result_high = predict_diabetes(high_risk_person)
print(f"👉 Result: {result_high}")

print(f"\nTesting Low Risk Input: {low_risk_person}")
result_low = predict_diabetes(low_risk_person)
print(f"👉 Result: {result_low}")


print("\n--- 🧪 TESTING OBESITY MODEL ---")
# 3. Obesity Test Case
# KEY FIX: Changed 'yes'/'no' to 1/0 for columns that don't have encoders
obesity_case = {
    'Gender': 'Male',        # Has encoder (Keep as text)
    'Age': 26,
    'Height': 1.75,
    'Weight': 95,
    'family_history_with_overweight': 1,  # 'yes' -> 1
    'FAVC': 1,               # 'yes' -> 1
    'FCVC': 2.0,
    'NCP': 3.0,
    'CAEC': 'Sometimes',     # Has encoder (Keep as text)
    'SMOKE': 0,              # 'no' -> 0
    'CH2O': 2.0,
    'SCC': 0,                # 'no' -> 0
    'FAF': 0.0,
    'TUE': 1.0,
    'CALC': 'Sometimes',     # Has encoder (Keep as text)
    'MTRANS': 'Public_Transportation' # Has encoder (Keep as text)
}

print(f"Testing Obesity Input: Weight 95kg, Height 1.75m")
result_obesity = predict_obesity(obesity_case)
print(f"👉 Result: {result_obesity}")