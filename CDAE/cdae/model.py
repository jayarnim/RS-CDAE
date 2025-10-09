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
            n_users (int): total number of users
            n_items (int): total number of items
            n_factors (int): latent dimension
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
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        x: torch.Tensor,
    ):
        """
        Training method

        Args:
            user_idx (torch.Tensor): user indices (B,)
            x (torch.Tensor): user's interactions (B, I)
        """
        return self.score(user_idx, x)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        x: torch.Tensor,
    ):
        """
        Evaluation method

        Args:
            user_idx (torch.Tensor): user indices (B,)
            x (torch.Tensor): user's interactions (B, I)
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
        h_user = self.user_embed(user_idx).squeeze(1)
        # 4. Combine: (B,K) + (B,K) -> (B,K)
        h = F.relu(h_user + h_hist)
        # 5. Reconstruction: (B,K) -> (B,I)
        x_hat = self.decoder(h)
        return x_hat

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        self.dropout = nn.Dropout(
            p=self.dropout,
        )
        self.user_embed = nn.Embedding(
            num_embeddings=self.n_users, 
            embedding_dim=self.n_factors,
        )
        self.encoder = nn.Linear(
            in_features=self.n_items, 
            out_features=self.n_factors,
        )
        self.decoder = nn.Linear(
            in_features=self.n_factors, 
            out_features=self.n_items,
        )