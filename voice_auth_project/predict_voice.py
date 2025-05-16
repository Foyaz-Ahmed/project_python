from extract_features import extract_mfcc
import joblib

def predict_voice(file_path):
    model = joblib.load('voice_model.pkl')  # Load the trained model
    features = extract_mfcc(file_path)  # Extract features from the test file
    result = model.predict([features])[0]
    return "Your Voice" if result == 1 else "Not Your Voice"