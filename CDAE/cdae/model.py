import torch
import torch.nn as nn
import torch.nn.functional as F


class Module(nn.Module):
    def __init__(
        self, 
        n_users: int, 
        n_items: int, 
        n_factors: int, 
        dropout: float=0.5,
    ):
        """
        PyTorch implementation of Collaborative Denoising AutoEncoder (CDAE)

        Args:
            n_users (int): total number of users (U)
            n_items (int): total number of items (I)
            n_factors (int): latent dimension (K)
            dropout (float): dropout rate (denoising)
        """
        super(Module, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.dropout = dropout

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        x: torch.Tensor,
    ):
        """
        user_idx: (B,)  - user indices (int)
        x: (B, I)       - binary vector (user's interactions)
        """
        return self.score(user_idx, x)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        x: torch.Tensor,
    ):
        """
        user_idx: (B,)  - user indices (int)
        x: (B, I)       - binary vector (user's interactions)
        """
        with torch.no_grad():
            logit = self.score(user_idx, x)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, x):
        return self.recon(user_idx, x)

    def recon(self, user_idx, x):
        # 1. Denoising: (B,I)
        x_denoised = self.dropout(x)
        # 2. Item encoding: (B,I) -> (B,K)
        h_hist = self.encoder(x_denoised)
        # 3. User embedding: (B,K)
        h_user = self.embed_user(user_idx).squeeze(1)
        # 4. Combine: (B,K) + (B,K) -> (B,K)
        h = F.relu(h_user + h_hist)
        # 5. Reconstruction: (B,K) -> (B,I)
        recon = self.decoder(h)
        return recon

    def _init_layers(self):
        # === Layers ===
        self.dropout = nn.Dropout(p=self.dropout)
        self.embed_user = nn.Embedding(self.n_users, self.n_factors)
        self.encoder = nn.Linear(self.n_items, self.n_factors)
        self.decoder = nn.Linear(self.n_factors, self.n_items)

        # === Initialization ===
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.normal_(self.embed_user.weight, std=0.01)