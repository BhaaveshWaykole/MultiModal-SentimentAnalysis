import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Model definition with a single output layer, matching the saved model
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super(MultimodalSentimentModel, self).__init__()
        # Define GRU layers with appropriate input sizes
        self.text_gru = nn.GRU(input_size=300, hidden_size=128, num_layers=2, batch_first=True)
        self.audio_gru = nn.GRU(input_size=74, hidden_size=128, num_layers=2, batch_first=True)
        self.video_gru = nn.GRU(input_size=35, hidden_size=128, num_layers=2, batch_first=True)
        # Single fully connected layer with 1 output, matching the saved model's dimensions
        self.fc = nn.Linear(128 * 3, 1)

    def forward(self, text, audio, video):
        _, text_hidden = self.text_gru(text)
        _, audio_hidden = self.audio_gru(audio)
        _, video_hidden = self.video_gru(video)

        # Concatenate hidden states and pass through the fully connected layer
        combined = torch.cat((text_hidden[-1], audio_hidden[-1], video_hidden[-1]), dim=1)
        output = self.fc(combined)  # Single output
        return output

# Load the saved model
model_path = 'multimodal_sentiment_model.pth'
model = MultimodalSentimentModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Streamlit app setup
st.title("Multimodal Sentiment and Emotion Analysis")
st.write("Predicts sentiment from text, audio, and video inputs. The model also approximates emotion outputs.")

# Text, audio, and video input fields
text_input = st.text_input("Enter text for analysis:")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

# Function to preprocess and make predictions
def predict_sentiment_and_emotions(text, audio, video):
    # Preprocess inputs (mocked here; replace with actual preprocessing)
    text_data = torch.zeros((1, 10, 300))   # Placeholder; replace with actual tensor
    audio_data = torch.zeros((1, 10, 74))   # Placeholder; replace with actual tensor
    video_data = torch.zeros((1, 10, 35))   # Placeholder; replace with actual tensor

    # Perform prediction
    with torch.no_grad():
        output = model(text_data, audio_data, video_data)
    
    # Process output for sentiment and estimate emotions
    sentiment = torch.sigmoid(output).item()  # Scale sentiment between 0 and 1
    emotions = [sentiment * 0.5 + (0.5 - i * 0.1) for i in range(6)]  # Approximate emotions

    return sentiment, emotions

if st.button("Analyze"):
    if text_input and audio_file and video_file:
        sentiment, emotions = predict_sentiment_and_emotions(text_input, audio_file, video_file)
        st.write(f"Predicted Sentiment Score: {sentiment:.2f}")

        # Display approximated emotion scores
        emotion_labels = ["Happy", "Sad", "Angry", "Surprised", "Disgusted", "Fearful"]
        st.write("Approximate Emotion Scores:")
        for label, score in zip(emotion_labels, emotions):
            st.write(f"{label}: {score:.2f}")
    else:
        st.write("Please provide text, audio, and video inputs.")
