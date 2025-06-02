# Architecture parameters for a transformer model
n_layers = 2
n_heads = 4
d_model = 128
d_head = 32
n_ctx = 64
act_fn = "gelu"
normalization_type = "LNPre"

# Attention configuration
attn_only = True  # include only attention layers, no MLPs
use_attn_result = True

# RoPE configuration
positional_embedding_type = "rotary"
rotary_base = 1000

# Training parameters
batch_size = 16
epochs = 10
learning_rate = 1e-3

# Paths
tokenizer_dir = "model/wordlevel_tokenizer"
data_path = "data/training_data.txt"
save_dir = "model/attn_only_rope"

# WandB config
use_wandb = True
project_name = "round_and_round_rope"
run_name = "attn_only_rope"

# Auto Circuit configuration
score_threshold = 4.5
