from functools import partial
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils
import torch
from utils.plot_head import imshow
from typing import Optional, Union, Literal
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------for VISUALIZATION-------------------------------------------------

def get_all_rope_acts(model, all_cache):
    """
    Get all the RoPE related activations from the cache.
    """
    rope_acts_q = []
    rope_acts_k = []
    pre_rope_acts_q = []
    pre_rope_acts_k = []
    for i in range(model.cfg.n_layers):
        rope_q = all_cache[utils.get_act_name("rot_q", f"{i}", "a")]
        rope_k = all_cache[utils.get_act_name("rot_k", f"{i}", "a")]
        pre_rope_q = all_cache[utils.get_act_name("q", f"{i}", "a")]
        pre_rope_k = all_cache[utils.get_act_name("k", f"{i}", "a")]
        rope_acts_q.append(rope_q)
        rope_acts_k.append(rope_k)
        pre_rope_acts_q.append(pre_rope_q)
        pre_rope_acts_k.append(pre_rope_k)
    return rope_acts_q, rope_acts_k, pre_rope_acts_q, pre_rope_acts_k

def plot_rope_freq_per_head_layer(rot_q: Tensor, rot_k: Tensor, mode: Union[Literal['heads, layers', 'per_layer']], layer_to_plot: int):
    """
    Visualizes the usage of RoPE (Rotary Positional Embedding) frequencies across attention heads and/or layers in a transformer model.
    Depending on the selected mode, this function computes and plots the mean norm of RoPE frequency components for the query and key tensors, either:
      - for each attention head in a specific layer (`mode="per_layer"`),
      - for each layer averaged over heads (`mode="layers"`),
      - or for each head averaged over layers (`mode="heads"`).

    Args:
        rot_q (Tensor): Query tensor of shape (n_layers, bsz, seq_len, n_heads, head_dim), containing RoPE-embedded queries.
        rot_k (Tensor): Key tensor of the same shape as `rot_q`, containing RoPE-embedded keys.
        mode (str): Visualization mode. One of:
            - "per_layer": Show frequency usage per head in a specific layer.
            - "layers": Show frequency usage per layer, averaged over heads.
            - "heads": Show frequency usage per head, averaged over layers.
        layer_to_plot (int): Index of the layer to visualize when `mode="per_layer"`. Ignored for other modes.
    
    Returns:
        Figure containing the mean norm of RoPE frequency components for rotated queries and keys.
    """
    def compute_freq_layers(tensor: Tensor):
        # Reshape last dim into (n_freqs, 2) where each pair is a complex rotation
        n_layers, bsz, seq_len, n_heads, head_dim = tensor.shape
        assert head_dim % 2 == 0, "Head dim must be even for RoPE frequency pairing"
        freq_tensor = tensor.reshape(n_layers, bsz, seq_len, n_heads, head_dim // 2, 2)
        freq_norms = np.linalg.norm(freq_tensor, axis=-1)  # n_layers, bsz, seq_len, n_heads, n_freqs
        return freq_norms.mean(axis=(1, 2, 3))  # n_layers, n_freqs
    
    def compute_freq_heads(tensor: Tensor):
        # Extract tensor for the specified layer
        n_layers, bsz, seq_len, n_heads, head_dim = tensor.shape
        assert head_dim % 2 == 0, "Head dim must be even for RoPE pairing"
        freq_tensor = tensor.reshape(n_layers, bsz, seq_len, n_heads, head_dim // 2, 2)
        freq_norms = np.linalg.norm(freq_tensor, axis=-1)  # n_layers, bsz, seq_len, n_heads, n_freqs
        return freq_norms.mean(axis=(0, 1, 2))  # mean over batch, seq, and heads -> n_layers, n_freqs
    
    def compute_freq_heads_per_layer(tensor: Tensor, layer: int):
        # Extract tensor for the specified layer
        tensor_l = tensor[layer]  # bsz, seq_len, n_heads, head_dim
        bsz, seq_len, n_heads, head_dim = tensor_l.shape
        assert head_dim % 2 == 0, "Head dim must be even for RoPE pairing"
        freq_tensor = tensor_l.reshape(bsz, seq_len, n_heads, head_dim // 2, 2)
        freq_norms = np.linalg.norm(freq_tensor, axis=-1)  # bsz, seq_len, n_heads, n_freqs
        return freq_norms.mean(axis=(0, 1))  # n_heads, n_freqs
    
    if mode=="per_layer" and layer_to_plot is not None: 
        q_heads = compute_freq_heads_per_layer(rot_q, layer_to_plot)
        k_heads = compute_freq_heads_per_layer(rot_k, layer_to_plot)
    if mode == "layers":
        q_heads = compute_freq_layers(rot_q)
        k_heads = compute_freq_layers(rot_k)
    elif mode == "heads":
        q_heads = compute_freq_heads(rot_q)
        k_heads = compute_freq_heads(rot_k)

    def plot_head_freqs(
    q_heads: torch.Tensor,
    k_heads: torch.Tensor,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    xaxis: str = "Attention Head",
    yaxis: str = "Frequencies",
    title_suffix: str = "",
    **imshow_kwargs
    ) -> None:
        """
        Plots query and key frequency norms for each head using your custom imshow setup.
        """
        if zmax is None:
            zmax = max(q_heads.max().item(), k_heads.max().item())
        if zmin is None:
            zmin = min(q_heads.min().item(), k_heads.min().item())

        # Plot query head frequencies
        imshow(
        q_heads.T,
        zmin=zmin,
        zmax=zmax,
        xaxis=xaxis,
        yaxis=yaxis,
        title=f"Query frequency usage {title_suffix}",
            **imshow_kwargs
        )
        imshow(
            k_heads.T,
            zmin=zmin,
            zmax=zmax,
            xaxis=xaxis,
            yaxis=yaxis,
            title=f"Key frequency usage {title_suffix}",
            **imshow_kwargs
        )
    color = "GnBu" # GnBu is used in the paper
    if mode=="per_layer" and layer_to_plot is not None: 
        # old plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        im0 = axes[0].imshow(q_heads.T, aspect='auto', origin='lower', cmap=color)
        axes[0].set_title(f"Query frequency usage (Layer {layer_to_plot})")
        axes[0].set_xlabel("Attention Head")
        axes[0].set_ylabel("Frequencies")
        fig.colorbar(im0, ax=axes[0], label="Mean norm")

        im1 = axes[1].imshow(k_heads.T, aspect='auto', origin='lower', cmap=color)
        axes[1].set_title(f"Key frequency usage (Layer {layer_to_plot})")
        axes[1].set_xlabel("Attention Head")
        axes[1].set_ylabel("Frequencies")
        fig.colorbar(im1, ax=axes[1], label="Mean norm")

        plt.tight_layout()
        plt.show()
    
        # plot_head_freqs(
        #     q_heads, k_heads,
        #     title_suffix=f"(Layer {layer})",
        #     color_continuous_scale="GnBu", # GnBu is used in the paper               
        # )

    elif mode == "layers":
        # old plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        im0 = axes[0].imshow(q_heads.T, aspect='auto', origin='lower', cmap=color)
        axes[0].set_title("Query frequency usage per layer")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Frequencies")
        fig.colorbar(im0, ax=axes[0], label="Mean norm")

        im1 = axes[1].imshow(k_heads.T, aspect='auto', origin='lower', cmap=color)
        axes[1].set_title("Key frequency usage per layer")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Frequencies")
        fig.colorbar(im1, ax=axes[1], label="Mean norm")

        plt.tight_layout()
        plt.show()

        # plot_head_freqs(
        #     q_heads, k_heads,
        #     xaxis="Layer",
        #     title_suffix=f"per layer",
        #     color_continuous_scale=color,               
        # )
    elif mode == "heads":
        # old plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        im0 = axes[0].imshow(q_heads.T, aspect='auto', origin='lower', cmap=color)
        axes[0].set_title("Query frequency usage per head")
        axes[0].set_xlabel("Head")
        axes[0].set_ylabel("Frequencies")
        fig.colorbar(im0, ax=axes[0], label="Mean norm")

        im1 = axes[1].imshow(k_heads.T, aspect='auto', origin='lower', cmap=color)
        axes[1].set_title("Key frequency usage per head")
        axes[1].set_xlabel("Head")
        axes[1].set_ylabel("Frequencies")
        fig.colorbar(im1, ax=axes[1], label="Mean norm")

        plt.tight_layout()
        plt.show()

        # plot_head_freqs(
        #     q_heads, k_heads,
        #     xaxis="Head",
        #     title_suffix=f"per head",
        #     color_continuous_scale=color,               
        # )

def extract_rope_frequency_usage(
    model: HookedTransformer,
    prompt: str,
    q_rot: Tensor,
    k_rot: Tensor,
    layer_idx: int,
    head_idx: int,
    q_or_k: str,
    show_freqs: bool = True
):
    """
    Visualize RoPE frequency usage per token (x-axis: tokens, y-axis: frequencies) frequencies for a specific attention head in a transformer model, across the tokens of a given prompt.
    
    Args:
        model (HookedTransformer): The transformer model instance, supporting tokenization and decoding.
        prompt (str): The input text prompt to analyze.
        q_rot (Tensor): The RoPE-rotated query tensor for the specified layer and head (shape: [seq_len, head_dim]).
        k_rot (Tensor): The RoPE-rotated key tensor for the specified layer and head (shape: [seq_len, head_dim]).
        layer_idx (int): The index of the transformer layer to analyze.
        head_idx (int): The index of the attention head within the specified layer.
        q_or_k (str): Whether to visualize 'q' (query) or 'k' (key) RoPE activations. Must be either "q" or "k".
        show_freqs (bool, optional): If True, displays individual frequency indices on the y-axis; if False, groups frequencies as "High" and "Low". Defaults to True.
    
    Returns:
        A heatmap where the x-axis corresponds to tokens in the prompt, the y-axis to RoPE frequency indices (or grouped frequencies), and the colorbar represents the L2 norm of the RoPE embedding for each frequency and token.
    """
    assert q_or_k in ("q", "k")

    toks = model.to_tokens(prompt)
    decoded_tokens = [model.tokenizer.decode([t]) for t in toks[0]]
    
    ## NOTE: maybe simplify this by using the model's cache directly
    # _, cache = model.run_with_cache(toks)

    # Extract RoPE activation
    # rope_name = f"blocks.{layer_idx}.attn.hook_rot_{q_or_k}"
    # rot_act = cache[rope_name]  # shape: (bsz, seq_len, n_heads, head_dim)
    # head_rot = rot_act[0, :, head_idx, :]  # shape: (seq_len, head_dim)

    head_rot = q_rot if q_or_k == "q" else k_rot
    
    # Convert to (seq_len, n_freqs, 2) and compute norm
    head_dim = head_rot.shape[-1]
    assert head_dim % 2 == 0
    n_freqs = head_dim // 2
    freq_reshaped = head_rot.view(-1, n_freqs, 2)
    freq_norms = torch.norm(freq_reshaped, dim=-1).T.cpu().numpy()  # shape: (n_freqs, seq_len)

    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(
        freq_norms,
        cmap="GnBu",
        xticklabels=decoded_tokens,
        yticklabels=np.arange(0, n_freqs),
    )
    if not show_freqs:
        ax.set_yticklabels([])
        
        ax.invert_yaxis()
        ax.set_yticks([0.9, n_freqs - int(0.1 * n_freqs )])  
        ax.set_yticks([4, freq_norms.shape[0] - 12])
        ax.set_yticklabels(["High Frequencies", "Low Frequencies"], rotation=90, fontsize=12)

    plt.title(f"{'Query' if q_or_k == 'q' else 'Key'} RoPE Frequency Usage (Layer {layer_idx} Head {head_idx})")
    plt.xlabel("Token")
    plt.ylabel("")  # NOTE: Remove "Frequency Index", maybe add a bool arg 

    plt.tight_layout()
    plt.show()


# ----------------------------for ABLATIONS-------------------------------------------------

def hook_save_pre_q(layer, save_arr):

    def save_pre_q(tensor, hook, save_arr):
        # tensor: (b, seq_len, d_model)
        save_arr.append(tensor.clone())
        return tensor

    return (f"blocks.{layer}.attn.hook_q", partial(save_pre_q, save_arr=save_arr))

def hook_save_pre_k(layer, save_arr):

    def save_pre_k(tensor, hook, save_arr):
        # tensor: (b, seq_len, d_model)
        save_arr.append(tensor.clone())
        return tensor

    return (f"blocks.{layer}.attn.hook_k", partial(save_pre_k, save_arr=save_arr))

def hook_save_rot_q(layer, save_arr):

    def save_rot_q(tensor, hook, save_arr):
        # tensor: (b, seq_len, d_model)
        save_arr.append(tensor.clone())
        return tensor

    return (f"blocks.{layer}.attn.hook_rot_q", partial(save_rot_q, save_arr=save_arr))

def hook_save_rot_k(layer, save_arr):

    def save_rot_k(tensor, hook, save_arr):
        # tensor: (b, seq_len, d_model)
        save_arr.append(tensor.clone())
        return tensor

    return (f"blocks.{layer}.attn.hook_rot_k", partial(save_rot_k, save_arr=save_arr))

def ablate_rope_features(
    q_post: Tensor,
    k_post: Tensor,
    q_pre: Tensor,
    k_pre: Tensor,
    ROT_FEAT: int,
    type: Union[Literal['single', 'all_but']]
):
    """
    Ablate specific rotary positional encoding (RoPE) features in rotated query and key tensors.
    The method of ablating RoPE features are by either zeroing out or replacing
    selected feature dimensions in the post-rotated query and key tensors. Two ablation modes are supported:
    - 'single': Ablate only the specified RoPE feature (dimensions [2i, 2i+1]).
    - 'all_but': Ablate all features except the specified one.

    Args:
        q_post (Tensor): Query tensor after RoPE application, shape (seq_len, n_heads, head_dim).
        k_post (Tensor): Key tensor after RoPE application, shape (seq_len, n_heads, head_dim).
        q_pre (Tensor): Query tensor before RoPE application, shape (seq_len, n_heads, head_dim).
        k_pre (Tensor): Key tensor before RoPE application, shape (seq_len, n_heads, head_dim).
        ROT_FEAT (int): Index of the RoPE feature to ablate or preserve.
        type (str): Ablation mode, either 'single' (ablate only ROT_FEAT) or 'all_but' (ablate all except ROT_FEAT).
        
        
    Returns:    
        dict: Dictionary with the following keys:
            - "orig": Tuple of (q_post, k_post), the original post-rotated tensors.
            - "zero": Tuple of ablated tensors where selected features are zeroed out.
            - "replace": Tuple of ablated tensors where selected features are replaced with pre-rotated values.
    """
    i = ROT_FEAT
    start = 2*i
    width = 2
    q_abl_zero = q_post.clone()
    q_abl_replace = q_post.clone()
    k_abl_zero = k_post.clone()
    k_abl_replace = k_post.clone()

    if type == 'single':
        q_abl_zero[:, start:start+width] = 0.0
        k_abl_zero[:, start:start+width] = 0.0
    
        q_abl_replace[:, start:start+width] = q_pre[:, start:start+width]
        k_abl_replace[:, start:start+width] = k_pre[:, start:start+width]

    elif type == "all_but":
        seq_len, n_heads, head_dim = q_pre.shape
        mask = torch.ones(head_dim, dtype=torch.bool)
        mask[start : start+width] = False  # now mask.sum() == 62
        
        # Zero-ablate ALL BUT that RoPE feature:
        q_abl_zero[:, :, mask] = 0.0          
        k_abl_zero[:, :, mask] = 0.0

        # Identity-ablate ALL BUT that RoPE feature (equivalent to NoPE):
        q_abl_replace[:, :, mask] = q_pre[:, :, mask]
        k_abl_replace[:, :, mask] = k_pre[:, :, mask]
    
    return {
      "orig": (q_post, k_post),
      "zero": (q_abl_zero, k_abl_zero),
      "replace": (q_abl_replace, k_abl_replace),
    }
