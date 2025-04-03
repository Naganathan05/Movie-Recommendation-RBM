import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, input_size),
            nn.Sigmoid()  # Output between 0-1 for normalized ratings
        )
    
    def forward(self, x):
        # Forward pass through encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def predict(self, x):
        """Predict ratings for a user based on their existing ratings
        
        Note: This matches the signature expected by the recommendation function
        """
        # Get the prediction
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            prediction = self.forward(x)
        return prediction