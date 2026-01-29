import sys
import json
import os

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the logic functions we created in ai_api.py
from ai_api import predict_diabetes, predict_obesity

# 1. Read data passed from Node.js
try:
    input_json = sys.argv[1]
    user_data = json.loads(input_json)
except Exception as e:
    print(json.dumps({"error": "Invalid JSON input", "details": str(e)}))
    sys.exit(1)

# 2. Run Predictions
# CRITICAL CHANGE: We do NOT manually extract data here anymore.
# We pass 'user_data' directly because predict_diabetes() and predict_obesity()
# now contain the logic to map React keys (e.g., 'veggieFreq') to Model inputs.

diabetes_result = predict_diabetes(user_data)
obesity_result = predict_obesity(user_data)

# 3. Combine and Print JSON
final_response = {
    "diabetes": diabetes_result,
    "obesity": obesity_result
}

print(json.dumps(final_response))