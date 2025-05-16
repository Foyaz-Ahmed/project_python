from extract_features import extract_mfcc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

your_samples = ['known_voices/fz.wav', 'known_voices/test_voice.wav'] #my samples
other_samples = ['known_voices/mithu.wav', 'known_voices/mithu1.wav']  #friends samples

X = []
y = []

# Extract features from my samples
for path in your_samples:
    if os.path.exists(path):
        X.append(extract_mfcc(path))
        y.append(1)

# Extract features from non-matching samples
for path in other_samples:
    if os.path.exists(path):
        X.append(extract_mfcc(path))
        y.append(0)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'voice_model.pkl')
print("✅ Model trained and saved.")