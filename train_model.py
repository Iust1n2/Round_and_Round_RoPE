import os
import json
import torch
import argparse
import importlib.util
from random import seed
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from transformers import PreTrainedTokenizerFast
from transformer_lens import HookedTransformer, HookedTransformerConfig

from utils.train_utils import (
    clean_config_dict,
    logit_diff_score,
    load_dataset,
    compute_accuracy,
)
from utils.circuit_discovery import (
    auto_circuit_experiment,
    get_real_edges,
    compute_circuit_overlap,
    plot_circuit_overlap_vs_accuracy,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
seed(SEED)
torch.manual_seed(SEED)


def main(params, use_wandb):
    tokens_seen = 0  # Global token counter
    tokenizer = PreTrainedTokenizerFast.from_pretrained(params.tokenizer_dir)
    vocab_size = tokenizer.vocab_size

    dataset = load_dataset(params.data_path, tokenizer)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    next_token_id = tokenizer.encode("next")[0]

    next_train_data = [sample for sample in train_data if next_token_id in sample]
    last_train_data = [sample for sample in train_data if next_token_id not in sample]

    def collate(batch):
        padded = pad_sequence(
            batch, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = padded.clone()
        return padded.to(DEVICE), labels.to(DEVICE)

    next_train_loader = DataLoader(
        next_train_data, batch_size=params.batch_size, shuffle=True, collate_fn=collate
    )
    last_train_loader = DataLoader(
        last_train_data, batch_size=params.batch_size, shuffle=True, collate_fn=collate
    )
    full_train_loader = DataLoader(
        train_data, batch_size=params.batch_size, shuffle=True, collate_fn=collate
    )

    val_loader = DataLoader(
        val_data, batch_size=params.batch_size, shuffle=False, collate_fn=collate
    )

    config = HookedTransformerConfig(
        n_layers=params.n_layers,
        n_heads=params.n_heads,
        d_model=params.d_model,
        d_head=params.d_head,
        d_mlp=params.d_mlp if hasattr(params, "d_mlp") else None,
        d_vocab=vocab_size,
        n_ctx=params.n_ctx,
        act_fn=params.act_fn,
        normalization_type=params.normalization_type,
        attn_only=params.attn_only,
        use_attn_result=params.use_attn_result,
        positional_embedding_type=params.positional_embedding_type,
        rotary_base=params.rotary_base if hasattr(params, "rotary_base") else None,
    )

    model = HookedTransformer(config).to(DEVICE)
    model.tokenizer = tokenizer
    model.set_use_attn_result(True)
    model.set_use_attn_in(True)
    model.set_use_split_qkv_input(True)

    if not params.attn_only:
        model.set_use_hook_mlp_in(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    if use_wandb:
        import wandb

        wandb.init(
            project=params.project_name,
            name=params.run_name,
            config={
                "epochs": params.epochs,
                "batch_size": params.batch_size,
                "lr": params.learning_rate,
                "tokenizer": params.tokenizer_dir,
                **clean_config_dict(model.cfg.to_dict()),
            },
        )

    accuracies = []
    circuit_overlap_logs = []

    print("Starting training...")
    for epoch in range(params.epochs):
        total_loss_A = total_loss_B = 0
        total_logit_diff_A = total_logit_diff_B = 0

        attribution_scores = {}

        model.train()
        for phase, train_loader in [
            ("A_last_token", last_train_loader),
            ("B_next_token", next_train_loader),
        ]:
            phase_loss = 0
            phase_logit_diff = 0

            for inputs, labels in train_loader:
                logits = model(inputs)
                logits_trimmed = logits[:, :-1]
                labels_trimmed = labels[:, 1:]

                flat_logits = logits_trimmed.reshape(-1, vocab_size)
                flat_labels = labels_trimmed.reshape(-1)

                loss = loss_fn(flat_logits, flat_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_token_count = (
                    (labels_trimmed != tokenizer.pad_token_id).sum().item()
                )
                tokens_seen += batch_token_count
                logit_diff = logit_diff_score(logits_trimmed, labels_trimmed)

                phase_loss += loss.item()
                phase_logit_diff += logit_diff

                if phase.startswith("A"):
                    total_loss_A = phase_loss
                    total_logit_diff_A = phase_logit_diff
                else:
                    total_loss_B = phase_loss
                    total_logit_diff_B = phase_logit_diff

                if use_wandb:
                    wandb.log(
                        {
                            "train/tokens_seen": tokens_seen,
                            f"train/loss_{phase}": loss.item(),
                            f"train/logit_diff_{phase}": logit_diff,
                        }
                    )

            if use_wandb:
                phase_id = "A" if phase.startswith("A") else "B"
                auto_circuit_save_path = (
                    f"auto_circuit_outputs_{params.run_name}_{phase_id}"
                )

                attribution_scores[phase_id] = auto_circuit_experiment(
                    model,
                    device=DEVICE,
                    save_path=auto_circuit_save_path,
                )

                last_path = f"{auto_circuit_save_path}/last.png"
                next_path = f"{auto_circuit_save_path}/next.png"
                if os.path.exists(last_path) and os.path.exists(next_path):
                    wandb.log(
                        {
                            f"auto_circuit/graph_last_after_{phase_id}": wandb.Image(
                                last_path, caption=f"After Phase {phase_id}"
                            ),
                            f"auto_circuit/graph_next_after_{phase_id}": wandb.Image(
                                next_path, caption=f"After Phase {phase_id}"
                            ),
                        }
                    )

            avg_phase_loss = phase_loss / len(train_loader)
            avg_phase_logit_diff = phase_logit_diff / len(train_loader)
            print(
                f"Epoch {epoch+1} | Phase {phase}: Loss = {avg_phase_loss:.4f}, LogitDiff = {avg_phase_logit_diff:.4f}"
            )

        train_acc = compute_accuracy(model, full_train_loader, tokenizer, DEVICE)
        accuracies.append(train_acc)
        print(f"Epoch {epoch+1} | Train Accuracy {train_acc:.4f}")

        if use_wandb:
            wandb.log(
                {
                    "train/epoch": epoch + 1,
                    "train/accuracy": train_acc,
                }
            )

            edges_A, remaining_edges_A = get_real_edges(
                model,
                attribution_scores["A"]["attribution_scores_last"],
                score_threshold=params.score_threshold,
                print_egdes=False,
                return_edges=True,
            )
            edges_B, remaining_edges_B = get_real_edges(
                model,
                attribution_scores["B"]["attribution_scores_next"],
                score_threshold=params.score_threshold,
                print_egdes=False,
                return_edges=True,
            )

            metrics = compute_circuit_overlap(
                edges_A,
                edges_B,
                attribution_scores_A=attribution_scores["A"]["attribution_scores_last"],
                attribution_scores_B=attribution_scores["B"]["attribution_scores_next"],
                print_diffs=False,
            )

            circuit_metrics = {
                "epoch": epoch + 1,
                "node_intersection": metrics["node_intersection"],
                "node_union": metrics["node_union"],
                "edge_intersection": metrics["edge_intersection"],
                "edge_union": metrics["edge_union"],
            }

            circuit_overlap_logs.append(circuit_metrics)

        # --- Validation ---
        model.eval()
        val_loss, val_logit_diff = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                logits = model(inputs)
                logits_trimmed = logits[:, :-1]
                labels_trimmed = labels[:, 1:]

                flat_logits = logits_trimmed.reshape(-1, vocab_size)
                flat_labels = labels_trimmed.reshape(-1)

                loss = loss_fn(flat_logits, flat_labels)
                val_loss += loss.item()
                val_logit_diff += logit_diff_score(logits_trimmed, labels_trimmed)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_logit_diff = val_logit_diff / len(val_loader)
        val_acc = compute_accuracy(model, val_loader, tokenizer, DEVICE)

        if use_wandb:
            wandb.log(
                {
                    "val/loss": avg_val_loss,
                    "val/accuracy": val_acc,
                    "val/logit_diff": avg_val_logit_diff,
                }
            )

        print(
            f"Epoch {epoch+1} | Val Loss {avg_val_loss:.4f}, Val Accuracy {val_acc:.4f}, Val LogitDiff {avg_val_logit_diff:.4f}"
        )

        avg_train_loss = (total_loss_A + total_loss_B) / (
            len(last_train_loader) + len(next_train_loader)
        )
        avg_train_logit_diff = (total_logit_diff_A + total_logit_diff_B) / (
            len(last_train_loader) + len(next_train_loader)
        )

        if use_wandb:
            wandb.log(
                {
                    "train/avg_loss": avg_train_loss,
                    "train/avg_logit_diff": avg_train_logit_diff,
                }
            )

        scheduler.step()

    if use_wandb:
        plot_circuit_overlap_vs_accuracy(
            epochs=range(params.epochs),
            accuracies=accuracies,
            circuit_overlap_logs=circuit_overlap_logs,
            save_dir=params.save_dir,
        )

        wandb.log(
            {
                "auto_circuit/node_overlap_vs_accuracy": wandb.Image(
                    f"{params.save_dir}/node_overlap_vs_accuracy.png",
                    caption="Node Overlap vs Accuracy",
                ),
                "auto_circuit/edge_overlap_vs_accuracy": wandb.Image(
                    f"{params.save_dir}/edge_overlap_vs_accuracy.png",
                    caption="Edge Overlap vs Accuracy",
                ),
            }
        )

    # Save directory
    Path(params.save_dir).mkdir(exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), os.path.join(params.save_dir, "model_weights.pth"))

    # Save config
    with open(os.path.join(params.save_dir, "config.json"), "w") as f:
        json.dump(clean_config_dict(model.cfg.to_dict()), f, indent=2)

    print(f"Model saved to {params.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/full_model_rope.py",
        help="Path to the model parameters configuration file.",
    )
    parser.add_argument(
        "-w",
        "--use_wandb",
        action="store_true",
        help="Enable logging to Weights & Biases.",
    )

    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

    spec = importlib.util.spec_from_file_location(
        config_path.replace(".py", "").replace("/", ".").replace("..", ""), config_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    main(module, args.use_wandb)
