import torch
import types
from typing import Callable
from sae_lens import StandardSAE
from collections import defaultdict
"""
Original encode/decode for StandardSAE
def encode(self, x: torch.Tensor) -> torch.Tensor:

        Encode the input tensor into the feature space.

        # Preprocess the SAE input (casting type, applying hooks, normalization)
        sae_in = self.process_sae_in(x)
        # Compute the pre-activation values
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        # Apply the activation function (e.g., ReLU, depending on config)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:

        Decode the feature activations back to the input space.
        Now, if hook_z reshaping is turned on, we reverse the flattening.

        # 1) linear transform
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        # 2) hook reconstruction
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        # 4) optional out-normalization (e.g. constant_norm_rescale)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        # 5) if hook_z is enabled, rearrange back to (..., n_heads, d_head).
        return self.reshape_fn_out(sae_out_pre, self.d_head)
"""


def monkeypatch_sae_decode_with_kernel(sae: StandardSAE, custom_sparse_matmul: Callable) -> StandardSAE:
    def write_decode_time_to_file(func_name, decode_time):
        with open(f"{func_name}_decode_times.txt", 'a') as f:
            f.write(f"{decode_time},")

    def custom_sparse_decode(self, feature_acts):
        # --- your custom sparse kernel here ---
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        sae_out_pre = custom_sparse_matmul(feature_acts, self.W_dec) + self.b_dec
        end.record()
        torch.cuda.synchronize()

        print("Decode time", start.elapsed_time(end), "ms")
        # This is hacky but just keeping this here for results
        name = getattr(custom_sparse_matmul, '__name__', repr(custom_sparse_matmul))
        write_decode_time_to_file(name, start.elapsed_time(end))

        # Then run the rest of the original decode pipeline
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    # orig_decode = sae.decode
    sae.decode = types.MethodType(custom_sparse_decode, sae)

    return sae

