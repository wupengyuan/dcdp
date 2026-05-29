import torch
import torch.nn as nn


class AsymmetricVAE(nn.Module):
    """Asymmetric VAE for normalized 4-step Push-T action chunks."""
    
    def __init__(self, action_dim=2, cond_dim=128, hidden_dim=128, latent_dim=2, action_horizon=4):
        super().__init__()
        self.action_horizon = action_horizon
        self.encoder = EncoderCNN(action_dim, hidden_dim, latent_dim, action_horizon)
        self.decoder = DecoderRNN(action_dim, latent_dim, cond_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, action_seq, temporal_cond):
        """
        action_seq: (B, 4, 2)
        temporal_cond: (B, 4, 128)
        """
        mu, logvar = self.encoder(action_seq)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, temporal_cond)
        return recon, mu, logvar


class EncoderCNN(nn.Module):
    """Encode a 4-step action chunk into a compact latent vector."""

    def __init__(self, action_dim=2, hidden_dim=128, latent_dim=2, action_horizon=4):
        super().__init__()
        self.action_horizon = action_horizon
        
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(action_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
        
        for _ in range(2):
            self.convs.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
        
        self.final_conv = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.fc_mu = nn.Linear(latent_dim * action_horizon, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * action_horizon, latent_dim)

    def forward(self, x):
        """
        x: (B, 4, 2) -> (B, 2, 4) for Conv1d
        """
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = torch.relu(conv(x))
        x = self.final_conv(x)
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class DecoderRNN(nn.Module):
    """Decode a latent action and 4-step dynamics context into actions."""

    def __init__(self, action_dim=2, latent_dim=2, cond_dim=128, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(latent_dim + cond_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, global_cond, temporal_cond):
        _, T, _ = temporal_cond.shape
        global_cond_expanded = global_cond.unsqueeze(1).repeat(1, T, 1)
        x = torch.cat([global_cond_expanded, temporal_cond], dim=-1)
        out, _ = self.gru(x)
        return self.fc_out(out)
