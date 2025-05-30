import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

# Input training data file
TRAINING_FILE = "data/training_data.txt"
TOKENIZER_OUTPUT_DIR = "model/wordlevel_tokenizer"

if not os.path.exists("./model"):
    os.makedirs("./model")

# Define special tokens
special_tokens = [
    "[UNK]",
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[BOS]",
    "[EOS]",  # BOS and EOS tokens added
]

# Step 1: Initialize the tokenizer
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Step 2: Set up trainer with special tokens
trainer = WordLevelTrainer(special_tokens=special_tokens, min_frequency=1)

# Step 3: Train the tokenizer
tokenizer.train(files=[TRAINING_FILE], trainer=trainer)

# Step 4: Wrap with Hugging Face tokenizer
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    bos_token="[BOS]",
    eos_token="[EOS]",
)

# Step 5: Save the tokenizer
wrapped_tokenizer.save_pretrained(TOKENIZER_OUTPUT_DIR)

print(f"Tokenizer with BOS/EOS trained and saved to '{TOKENIZER_OUTPUT_DIR}'")
