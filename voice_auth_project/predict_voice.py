from extract_features import extract_mfcc
import joblib

def predict_voice(file_path):
    model = joblib.load('voice_model.pkl')  # Load the trained model
    features = extract_mfcc(file_path)  # Extract features from the test file
    proba = model.predict_proba([features])[0]
    print("Probabilities:", proba)

    if len(proba) == 2:
        confidence = proba[1]  # Your voice
        result = "Your Voice" if confidence >= 0.5 else "Not Your Voice"
        percentage = round(confidence * 100, 2)
        return f"{result} (Confidence: {percentage}%)"
    else:
        # Handle the case when only one class exists
        result = model.predict([features])[0]
        return "Your Voice" if result == 1 else "Not Your Voice"