import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import random
from midiutil import MIDIFile
import os

CHORD_PROGRESSIONS = {
    1: [[60, 64, 67], [67, 71, 74], [69, 72, 76], [65, 69, 72]],
    2: [[60, 64, 67], [69, 72, 76], [62, 65, 69], [67, 71, 74]],
    3: [[60, 64, 67], [65, 69, 72], [69, 72, 76], [67, 71, 74]],
    4: [[60, 64, 67], [65, 69, 72], [67, 71, 74]],
    5: [[60, 64, 67], [67, 71, 74], [65, 69, 72], [69, 72, 76]],
    6: [[62, 65, 69], [67, 71, 74], [60, 64, 67]],
}

class ClimateLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=1):
        super(ClimateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def prepare_climate_data(df):
    df_clean = df.dropna(subset=['LandAverageTemperature'])
    sequence_data = []
    for _, row in df_clean.iterrows():
        temp = row['LandAverageTemperature']
        temp_normalized = (temp + 5) / 25
        progression_idx = min(6, max(1, int(temp_normalized * 6) + 1))
        progression = CHORD_PROGRESSIONS.get(progression_idx, CHORD_PROGRESSIONS[3])
        for chord in progression:
            for note in chord:
                sequence_data.append([note, temp])
    return np.array(sequence_data, dtype=np.float32)

def create_sequences(data, seq_length=16):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)

def train_lstm_model(X, y, epochs=20, batch_size=32):
    model = ClimateLSTM()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    return model

def generate_music_sequence(model, start_seq, length=100):
    model.eval()
    generated = []
    with torch.no_grad():
        inp = torch.tensor(start_seq, dtype=torch.float32).unsqueeze(0)
        for _ in range(length):
            next_note = model(inp).item()
            temp = inp[0, -1, 1].item()
            next_pair = [next_note, temp]
            generated.append(next_pair)
            new_input = torch.cat([inp[0, 1:], torch.tensor([next_pair])]).unsqueeze(0)
            inp = new_input
    return generated

def create_midi_file(note_sequence, filename="generated_climate_music.mid", tempo=120):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)
    for i, note in enumerate(note_sequence):
        note = max(21, min(108, int(round(note))))
        midi.addNote(0, 0, note, i * 0.5, 0.5, 100)
    with open(filename, "wb") as f:
        midi.writeFile(f)
    print(f"MIDI file saved as: {filename}")

def generate_climate_music(df, output_file="generated_climate_music.mid"):
    try:
        print("Preparing climate data...")
        data_array = prepare_climate_data(df)
        if len(data_array) == 0:
            print("Error: No valid data found")
            return False
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_array)
        X_seq, y_seq = create_sequences(data_scaled, seq_length=16)
        if len(X_seq) == 0:
            print("Error: Not enough data to create sequences")
            return False
        print(f"Training LSTM model with {len(X_seq)} sequences...")
        model = train_lstm_model(X_seq, y_seq, epochs=15)
        print("Generating music sequence...")
        start_seq = X_seq[0]
        generated = generate_music_sequence(model, start_seq, length=80)
        generated_arr = scaler.inverse_transform(np.array(generated))
        note_sequence = generated_arr[:, 0]
        print("Creating MIDI file...")
        create_midi_file(note_sequence, output_file)
        print("Music generation completed successfully!")
        return True
    except Exception as e:
        print(f"Error during music generation: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        df = pd.read_csv("sample_climate_data.csv")
        success = generate_climate_music(df)
        if success:
            print("Climate music generated successfully!")
        else:
            print("Failed to generate climate music.")
    except FileNotFoundError:
        print("Climate data file not found. Please ensure the data file exists.")
