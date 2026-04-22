import pickle
import xgboost as xgb
import os

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(current_dir, "xgboost_crop_model.pkl")
json_path = os.path.join(current_dir, "crop_recommendation_model.json")

# 1. Load the pickle
with open(pkl_path, "rb") as file:
    model = pickle.load(file)

# 2. Extract and save the Booster (the core model)
try:
    if hasattr(model, "get_booster"):
        # If it's the Scikit-Learn wrapper (XGBClassifier)
        booster = model.get_booster()
        booster.save_model(json_path)
    else:
        # If it's already a raw Booster
        model.save_model(json_path)
    print(f"✅ Success! Model saved to: {json_path}")
except Exception as e:
    print(f"❌ Failed to save: {e}")