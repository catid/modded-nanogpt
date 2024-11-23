import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryLayer(nn.Module):
    def __init__(self, input_dim, output_dim, K, tau, t=1.0):
        """
        MemoryLayer replaces a linear layer using hash-based memory lookup.

        Args:
            input_dim (int): Dimensionality of the input embeddings.
            output_dim (int): Dimensionality of the output embeddings.
            K (int): Number of chunks to split the input embedding into.
            tau (int): Bit width of each chunk after binarization.
            t (float): Temperature parameter for computing p(z_k).
        """
        super(MemoryLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = K  # Number of chunks
        self.tau = tau  # Bit width of each chunk
        self.t = t  # Temperature parameter

        assert input_dim % K == 0, "Input dimension must be divisible by K"
        self.chunk_size = input_dim // K

        assert output_dim % K == 0, "Output dimension must be divisible by K"
        self.output_chunk_size = output_dim // K

        # Initialize K hash tables as learnable embedding layers
        # Each hash table maps from 2^tau possible hash codes to output_chunk_size-dimensional vectors
        self.hash_tables = nn.ModuleList([
            nn.Embedding(2 ** self.tau, self.output_chunk_size)
            for _ in range(K)
        ])

    def forward(self, x):
        """
        Forward pass of the MemoryLayer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, output_dim].
        """
        batch_size, seq_len, _ = x.shape

        # Split the input tensor into K chunks along the last dimension
        x_chunks = x.split(self.chunk_size, dim=-1)  # Each chunk: [batch_size, seq_len, chunk_size]

        # Initialize the output tensor
        y_hat = torch.zeros(batch_size, seq_len, self.output_dim, device=x.device, dtype=x.dtype)

        # Process each chunk separately
        for k in range(self.K):
            z_k = x_chunks[k]  # Chunk k: [batch_size, seq_len, chunk_size], where chunk_size = tau

            # Compute the sign of z_k to obtain s_k ∈ {-1, +1}
            s_k = torch.sign(z_k)
            s_k[s_k == 0] = 1  # Map zeros to +1 to avoid zero hash codes

            # Convert s_k to binary representation: s_k_bin ∈ {0, 1}
            s_k_bin = (s_k + 1) // 2  # -1 -> 0, +1 -> 1

            # Compute the hash code h(z_k) by converting binary to integer
            exponents = 2 ** torch.arange(self.tau - 1, -1, -1, device=x.device, dtype=torch.long)
            h_zk = torch.sum(s_k_bin.long() * exponents.view(1, 1, -1), dim=-1)  # Shape: [batch_size, seq_len]

            # Retrieve the corresponding vectors from the hash table T_k using h(z_k)
            T_k = self.hash_tables[k]  # Hash table k
            retrieved = T_k(h_zk)  # Shape: [batch_size, seq_len, output_chunk_size]

            # Compute the weighting factor p(z_k)
            # p(z_k) = ∏_{i=0}^{τ-1} sigmoid(2 * z_k_i / t)
            log_p_zk = F.logsigmoid(2 * z_k / self.t).sum(dim=-1)  # Shape: [batch_size, seq_len]
            p_zk = torch.exp(log_p_zk)  # Shape: [batch_size, seq_len]

            # Apply the weighting to the retrieved vectors
            p_zk = p_zk.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
            weighted_retrieved = retrieved * p_zk  # Shape: [batch_size, seq_len, output_chunk_size]

            # Aggregate the weighted retrieved vectors into the output tensor
            start_idx = k * self.output_chunk_size
            end_idx = (k + 1) * self.output_chunk_size
            y_hat[:, :, start_idx:end_idx] += weighted_retrieved

        return y_hat

if __name__ == '__main__':
    # Define dimensions
    input_dim = 512  # Example input dimension
    output_dim = 512  # Example output dimension
    K = 64  # Number of chunks
    tau = 8  # Bit width per chunk

    batch_size = 4
    seq_len = 512

    # Create a MemoryLayer instance
    memory_layer = MemoryLayer(input_dim=input_dim, output_dim=output_dim, K=K, tau=tau)

    # Input tensor: [batch_size, seq_len, input_dim]
    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    y_hat = memory_layer(x)

