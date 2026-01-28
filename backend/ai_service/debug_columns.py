import joblib
import os

# Define path to models
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '..', '..', 'models')

print(f"📂 checking models in: {model_dir}")

try:
    # Load the Obesity Encoders
    encoders = joblib.load(os.path.join(model_dir, 'obesity_encoders.pkl'))
    
    print("\n✅ THE OBESITY MODEL EXPECTS THESE EXACT TEXT COLUMNS:")
    print("-------------------------------------------------------")
    for col in encoders.keys():
        print(f"'{col}'")
    print("-------------------------------------------------------")
    print("👉 Make sure your input dictionary in verify_models.py matches these EXACTLY.")

except FileNotFoundError:
    print("❌ Error: Could not find 'obesity_encoders.pkl'. Check your models folder.")
except Exception as e:
    print(f"❌ Error: {e}")