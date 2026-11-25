import torch
import torch.nn as nn

class WrappedOATS:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", activation_tokens=0, calibration_sequences=128):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name
        self.activation_tokens = activation_tokens
        self.tokens_per_sequence = max(1, (activation_tokens + calibration_sequences - 1) // calibration_sequences)
        self.activation_samples = []
        self.sampled_tokens = 0

    def add_batch(self, inp, out):
        if self.activation_tokens and self.sampled_tokens < self.activation_tokens:
            sequences = inp.unsqueeze(0) if inp.ndim == 2 else inp
            for sequence in sequences:
                count = min(self.tokens_per_sequence, self.activation_tokens - self.sampled_tokens)
                if count <= 0:
                    break
                indices = torch.linspace(0, sequence.shape[0] - 1, count, device=sequence.device).round().long()
                self.activation_samples.append(sequence.detach().index_select(0, indices).cpu())
                self.sampled_tokens += count
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp.clone().cpu(), p=2, dim=1) ** 2  / self.nsamples
