import librosa
import librosa.display
import soundfile as sf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import IPython.display as ipd

# Placeholder for data loading - replace with my actual data
# Assuming you have labeled audio files for different emergency vehicles
# and store them in directories named after vehicle type (e.g., "ambulance", "firetruck", etc.)

def load_audio_data(data_dir):
    audio_files = []
    labels = []
    # Replace with your data loading logic
    return audio_files, labels

# Function to extract Mel-spectrogram features from audio
def extract_mel_features(audio_file):
    y, sr = librosa.load(audio_file)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db

# Placeholder for real-time audio input - needs proper implementation
# This is just a simulation for demonstration

def process_audio(audio_file):

  # Extract Mel-spectrogram features
  mel_features = extract_mel_features(audio_file)

  # Flatten the features for the classifier
  mel_features_flattened = mel_features.flatten()

  # Predict the vehicle type
  predicted_class = model.predict([mel_features_flattened])
  return predicted_class

# Main execution flow
if __name__ == "__main__":
  # Load audio files and labels
  audio_files, labels = load_audio_data("C:\Storage\Python Scripts\emergency_vehicle_sounds\sounds")  # Replace with your data directory

  # Extract Mel-spectrogram features from all audio files
  features = []
  for file in audio_files:
      features.append(extract_mel_features(file).flatten()) # Flatten the Mel-spectrogram
  
  # Train-test split
  X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
  
  # Create and train a classifier (RandomForestClassifier example)
  model = RandomForestClassifier()
  model.fit(X_train, y_train)

  # Example usage with simulated real-time input
  # Replace this with actual real-time audio input and processing
  test_audio_files = ["C:\Storage\Python Scripts\emergency_vehicle_sounds\sound_1.wav", "C:\Storage\Python Scripts\emergency_vehicle_sounds\sound_401.wav"] # Example audio files for testing


  for file in test_audio_files:
    predicted_vehicle = process_audio(file) #Replace with your real time audio stream

    print(f"Predicted vehicle type for {file}: {predicted_vehicle}")

  # You would integrate this into a continuous loop for real-time processing
  # In the real-time scenario, instead of loading files, you'd acquire audio data from a microphone input.
  # Install pyaudio: !apt-get install portaudio19-dev; !pip install pyaudio

