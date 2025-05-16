from predict_voice import predict_voice

print("🎤 Voice Verification")
file_path = 'known_voices/test_voice.wav'
result = predict_voice(file_path)
print(f"Result: {result}")