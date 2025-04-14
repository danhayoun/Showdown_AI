import torch
import torch.nn as nn 


class TypeAutoencoder(nn.Module):
    def __init__(self, encoded_size=4):  # taille compressée ici
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(17, 12),
            nn.ReLU(),
            nn.Linear(12, encoded_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_size, 12),
            nn.ReLU(),
            nn.Linear(12, 17),
            nn.Sigmoid()  # si tu normalises les entrées
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded