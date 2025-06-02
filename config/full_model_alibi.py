# Architecture parameters for a transformer model
n_layers = 2
n_heads = 4
d_model = 128
d_head = 32
d_mlp = 512
n_ctx = 64
act_fn = "gelu"
normalization_type = "LNPre"

# Attention configuration
attn_only = False  # include both attention and MLPs
use_attn_result = True

# RoPE configuration
positional_embedding_type = "alibi"

# Training parameters
batch_size = 16
epochs = 10
learning_rate = 1e-3

# Paths
tokenizer_dir = "model/wordlevel_tokenizer"
data_path = "data/training_data.txt"
save_dir = "model/full_model_alibi"

# WandB config
use_wandb = True
project_name = "round_and_round_rope"
run_name = "full_model_alibi"

# Auto Circuit configuration
score_threshold = 3.5
