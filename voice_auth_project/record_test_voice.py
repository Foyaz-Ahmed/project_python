import sounddevice as sd
from scipy.io.wavfile import write

duration = 5
sample_rate = 44100
print("Recording will start in 2 seconds. Get ready...")

sd.sleep(3000) 

print("Recording...")

recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()

write('known_voices/test_voice.wav', sample_rate, recording)

print("Recording saved as test_voice.wav")


