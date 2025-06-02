import os
import pandas as pd
from typing import Optional, List, Dict, Literal, Union

import torch 
from transformer_lens import HookedTransformer

from auto_circuit.data import load_datasets_from_json
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores, Edge, Node, AblationType
from auto_circuit.visualize import node_name
from auto_circuit.utils.graph_utils import patchable_model, patch_mode
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.tensor_ops import (
    correct_answer_proportion, 
    correct_answer_greater_than_incorrect_proportion, 
    batch_avg_answer_diff, 
    batch_answer_diff_percents, 
    correct_answer_greater_than_incorrect_proportion
)

from PIL import Image
import matplotlib.pyplot as plt
import plotly.io as pio
import os


pio.renderers.default = "svg"


def auto_circuit_experiment(
    model: HookedTransformer, device: str = "cuda", score_threshold: int = None, save_path: str = None
) -> None:
    """
    Runs a circuit discovery experiment using **Edge Patching** with default config (Resample Ablation and slicing logits on the last position) on a given `HookedTransformer` model.
    The function computes attribution scores for both "last" and "next" datasets, visualizes the resulting graphs, and saves the figures.
    Args:
        model (HookedTransformer): The transformer model to analyze. Must be an instance of HookedTransformer.
        score_threshold (int, optional): The minimum absolute value of the score for an edge to be be kept in the circuit visualzation. Defaults to None.
        device (str, optional): The device to run computations on (e.g., "cuda" or "cpu"). Defaults to "cuda".
        save_path (str, optional): The directory path where the resulting images will be saved. If None, images are saved in the project root.

    Note:
        The data used for the patching experiment should not be passed from the dataloaders used in training.

    TODO:
        - Add implementation for tokenwise circuits.

    """
    assert isinstance(
        model, HookedTransformer
    ), "Model must be an instance of HookedTransformer"
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # path to the JSON datasets
    base_dir = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.abspath(os.path.join(base_dir, ".."))
    path_last = repo_path_to_abs_path(
        os.path.join(base_dir, "data/succession_augmented_last_big.json")
    )
    path_next = repo_path_to_abs_path(
        os.path.join(base_dir, "data/succession_augmented_next_big.json")
    )

    train_loader_last, _ = load_datasets_from_json(
        model=model,
        path=path_last,
        device=device,
        prepend_bos=False,
        tail_divergence=False,
        batch_size=32,
        train_test_size=(256, 256),
    )
    train_loader_next, _ = load_datasets_from_json(
        model=model,
        path=path_next,
        device=device,
        prepend_bos=False,
        tail_divergence=False,
        batch_size=32,
        train_test_size=(256, 256),
    )

    auto_model = patchable_model(
        model,
        factorized=True,
        slice_output="last_seq",
        separate_qkv=True,
        device=device,
    )

    attribution_scores_last: PruneScores = mask_gradient_prune_scores(
        model=auto_model,
        dataloader=train_loader_last,
        official_edges=None,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    )

    attribution_scores_next: PruneScores = mask_gradient_prune_scores(
        model=auto_model,
        dataloader=train_loader_next,
        official_edges=None,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    )

    fig_last = draw_seq_graph(
        auto_model,
        attribution_scores_last,
        score_threshold=score_threshold,
        layer_spacing=True,
        orientation="v",
        display_ipython=False,
    )
    fig_next = draw_seq_graph(
        auto_model,
        attribution_scores_next,
        score_threshold=score_threshold,
        layer_spacing=True,
        orientation="v",
        display_ipython=False,
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    last_img_path = f"{save_path}/last.png"
    next_img_path = f"{save_path}/next.png"

    fig_last.write_image(last_img_path, scale=1)
    fig_next.write_image(next_img_path, scale=1)

    return {
        "attribution_scores_last": attribution_scores_last,
        "attribution_scores_next": attribution_scores_next,
    }


def ablation_experiment(
    model: HookedTransformer,
    device: str,
    phase: Literal['A', 'B'],
    edges_sorted: List[Edge],
    ablation_type: Literal['Zero, Resample'] = None,
    how_many: Optional[int] = None,
):
    """
    Runs an ablation experiment on a given HookedTransformer model by patching specified edges and evaluating the effect on model performance, compared with the un-patched model.

    Args:
        model (HookedTransformer): The transformer model to be ablated.
        device (str): The device to run computations on (e.g., 'cpu' or 'cuda').
        phase (Literal['A', 'B']): Phase of the experiment, either 'A' or 'B'.
        edges_sorted (List[Edge]): List of edges sorted by attribution score (descending).
        ablation_type (Literal['Zero', 'Resample'], optional): Type of ablation to perform. 
            'Zero' sets activations to zero, 'Resample' replaces activations with resampled values.
        how_many (Optional[int], optional): Number of top edges (by attribution) to keep unpatched; all others are patched. If None, all edges are patched.
    Returns:
        dict: Dictionary containing ablation metrics:
            - "correct_answer_proportion": Proportion of correct answers after ablation.
            - "correct_answer_greater_than_incorrect_proportion": Proportion where correct answer logits exceed incorrect ones.
            - "batch_avg_answer_diff": Average difference between correct and incorrect answer logits (ablated).
            - "batch_avg_answer_diff_clean": Average difference between correct and incorrect answer logits (clean/original model).
    
    Raises:
        AssertionError: 
            - If the model is not an instance of HookedTransformer or if the top edge is not found in the model's edge dictionary.
            - If `ablation_type` is not one of the specified types.
            - `how_many` must be a non-negative integer > 1 else it errors.
    """
    assert isinstance(
        model, HookedTransformer
    ), "Model must be an instance of HookedTransformer"
 
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # path to the JSON datasets
    base_dir = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.abspath(os.path.join(base_dir, ".."))
    path_last = repo_path_to_abs_path(
        os.path.join(base_dir, "data/succession_augmented_last_big.json")
    )
    path_next = repo_path_to_abs_path(
        os.path.join(base_dir, "data/succession_augmented_next_big.json")
    )
    if phase == 'A':
        train_loader, test_loader = load_datasets_from_json(
            model=model,
            path=path_last,
            device=device,
            prepend_bos=False,
            tail_divergence=False,
            batch_size=32,
            train_test_size=(256, 256),
        )
    elif phase == 'B':
        train_loader, test_loader = load_datasets_from_json(
            model=model,
            path=path_next,
            device=device,
            prepend_bos=False,
            tail_divergence=False,
            batch_size=32,
            train_test_size=(256, 256),
        )
    
    auto_model = patchable_model(
        model,
        factorized=True,
        slice_output="last_seq",
        separate_qkv=True,
        kv_caches=(train_loader.kv_cache, test_loader.kv_cache), # this is needed else, patching will not work
        device=device,
    )

    attribution_scores: PruneScores = mask_gradient_prune_scores(
        model=auto_model,
        dataloader=train_loader,
        official_edges=None,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    )

    for batch in test_loader:
        toks = batch.clean
        answers = batch.answers
        wrong_answers = batch.wrong_answers
        answers = [(answers[i], wrong_answers[i]) for i in range(len(answers))]
        answers = torch.tensor(answers, dtype=torch.long)
        answers = answers.to(device)

    if ablation_type == 'Zero':
        ablations = src_ablations(auto_model, toks, AblationType.ZERO)

    elif ablation_type == 'Resample':
        ablations = src_ablations(auto_model, toks, AblationType.RESAMPLE)
        
    # quick sanity check if an edge is present
    all_edge_strs = [str(edge) for edge in edges_sorted] # these are desceonding order by attribution score
    edge_str_to_obj = {str(edge): edge for edge in edges_sorted}

    found = any(
        all_edge_strs[0] in subdict
            for subdict in auto_model.edge_name_dict.values()
        )
    assert found, f"Edge '{all_edge_strs[0]}' not found in edge_name_dict"

    # patch all edges except the ones with the highest attribution scores, specified by `how_many`
    if how_many is not None:
        remaining_edge_strs = all_edge_strs[:how_many]
        patch_edges = [e for e in all_edge_strs if e not in remaining_edge_strs]

        remaining_edges_with_scores = []
        patched_edges_with_scores = []

        for edge_str in remaining_edge_strs:
            edge = edge_str_to_obj[edge_str]
            score = attribution_scores[edge.dest.module_name][edge.patch_idx].item()
            remaining_edges_with_scores.append((edge_str, score))

        for edge_str in patch_edges:
            edge = edge_str_to_obj[edge_str]
            score = attribution_scores[edge.dest.module_name][edge.patch_idx].item()
            patched_edges_with_scores.append((edge_str, score))
    
    with patch_mode(auto_model, ablations, patch_edges):
        patched_out = auto_model(toks)

    for batch in test_loader:
        clean_out = model(batch.clean)
    
    ablation_metrics = {
        "remaining_edges_with_scores": remaining_edges_with_scores if how_many is not None else None,
        "patched_edges_with_scores": patched_edges_with_scores if how_many is not None else None,
        "correct_answer_proportion": correct_answer_proportion(
            patched_out[:, -1, :], batch),
        "correct_answer_greater_than_incorrect_proportion": correct_answer_greater_than_incorrect_proportion(
            patched_out[:, -1, :], batch),
        "batch_avg_answer_diff": batch_avg_answer_diff(
            patched_out[:, -1, :], batch),
        "batch_avg_answer_diff_clean": batch_avg_answer_diff(
            clean_out[:, -1, :], batch),
    }   
    return ablation_metrics


def get_real_edges(
    model: HookedTransformer,
    attribution_scores: PruneScores,
    score_threshold: int,
    print_egdes: bool = False,
    return_edges: bool = False,
) -> None:
    """
    Computes the number of edges in the circuit with edge attribution scores above a given threshold.
    If `print_egdes` is True, prints the remaining edges with their respective attribution scores.

    Args:
        auto_model (PatchableModel): The model containing the edge dictionary.
        attribution_scores (PruneScores): A mapping of attribution scores for each edge.
        score_threshold (int): The minimum absolute value of the score for an edge to be considered "real".
        print_egdes (bool, optional): If True, prints details of the remaining edges. Defaults to False.
    """
    real_edges = []

    auto_model = patchable_model(
        model,
        factorized=True,
        slice_output="last_seq",
        separate_qkv=True,
        device="cuda",
    )

    intervals = {list(auto_model.edge_dict.keys())[0]: (0, 1)}
    for seq_idx, _ in intervals.items():
        edge_set = set(auto_model.edge_dict[seq_idx])

    print(attribution_scores.keys())

    for edge in edge_set:
        edge_score = attribution_scores[edge.dest.module_name][edge.patch_idx].item()

        if abs(edge_score) < score_threshold:
            continue

        real_edges.append(edge)

    real_edges_sorted = sorted(
        real_edges,
        key=lambda e: attribution_scores[e.dest.module_name][e.patch_idx].item(),
        reverse=True,
    )

    print(
        f"No. of remaining edges with |score| ≥ {score_threshold}: {len(real_edges_sorted)}"
    )

    if print_egdes:
        for e in real_edges_sorted:
            s = attribution_scores[e.dest.module_name][e.patch_idx].item()
            print(f"  • {e.src.name} → {e.dest.name}:   score={s:.2f}")

    if return_edges:
        return real_edges_sorted, len(real_edges_sorted)


def compute_circuit_overlap(
    edges_A,
    edges_B,
    attribution_scores_A=None,
    attribution_scores_B=None,
    print_diffs: bool = False,
):
    """
    Given two lists of Edge objects from circuits A and B, optionally with their
    attribution_scores mappings, compute overlap metrics and, if requested, show
    which edges/nodes are unique to A, unique to B, or in the intersection.

    Args:
        edges_A (List[Edge]): List of Edge objects from circuit A.
        edges_B (List[Edge]): List of Edge objects from circuit B.
        attribution_scores_A (dict or None): Mapping for A's attribution scores,
            i.e. attribution_scores_A[module_name][patch_idx] → tensor. If provided,
            scores will be printed for edges in A\B or intersection.
        attribution_scores_B (dict or None): Same mapping for circuit B; used
            when printing edges in B\A or intersection.
        print_diffs (bool): If True, print which edges and nodes are in A\B, B\A,
            and in the intersection, along with each edge’s attribution score.

    Returns:
        dict: {
            "edge_intersection": int,
            "edge_union":        int,
            "edge_jaccard":    float,
            "node_intersection": int,
            "node_union":        int,
            "node_jaccard":    float,
        }
    Note:
        The Jaccard index for edges and nodes between two circuits is computed as |A ∩ B| / |A ∪ B|.
    """

    # Helper to create a unique key for each edge
    def edge_key(e):
        return (e.src.name, e.dest.name, e.patch_idx)

    # Build sets of edge‐keys for A and B
    set_A = {edge_key(e) for e in edges_A}
    set_B = {edge_key(e) for e in edges_B}

    # Also keep mapping from key → edge object, for printing
    dict_A = {edge_key(e): e for e in edges_A}
    dict_B = {edge_key(e): e for e in edges_B}

    # Intersection and union of keys
    inter_edges = set_A & set_B
    union_edges = set_A | set_B

    edge_intersection = len(inter_edges)
    edge_union = len(union_edges)
    edge_jaccard = (edge_intersection / edge_union) if edge_union > 0 else 1.0

    # Build node sets from edges
    nodes_A = {e.src.name for e in edges_A} | {e.dest.name for e in edges_A}
    nodes_B = {e.src.name for e in edges_B} | {e.dest.name for e in edges_B}

    inter_nodes = nodes_A & nodes_B
    union_nodes = nodes_A | nodes_B

    node_intersection = len(inter_nodes)
    node_union = len(union_nodes)
    node_jaccard = (node_intersection / node_union) if node_union > 0 else 1.0

    # If requested, print A\B, B\A, and intersection details
    if print_diffs:
        only_A_keys = set_A - set_B
        only_B_keys = set_B - set_A

        print("=== Edges only in A \\ B ===")
        if only_A_keys:
            for k in sorted(only_A_keys):
                e = dict_A[k]
                if attribution_scores_A is not None:
                    score = attribution_scores_A[e.dest.module_name][e.patch_idx].item()
                    print(
                        f"  • {e.src.name} → {e.dest.name}  attr_score(A)= {score:.2f}"
                    )
                else:
                    print(f"  • {e.src.name} → {e.dest.name}")
        else:
            print("  (none)")

        print("\n=== Edges only in B \\ A ===")
        if only_B_keys:
            for k in sorted(only_B_keys):
                e = dict_B[k]
                if attribution_scores_B is not None:
                    score = attribution_scores_B[e.dest.module_name][e.patch_idx].item()
                    print(
                        f"  • {e.src.name} → {e.dest.name}  attr_score(B)= {score:.2f}"
                    )
                else:
                    print(f"  • {e.src.name} → {e.dest.name}")
        else:
            print("  (none)")

        print("\n=== Edges in A ∩ B ===")
        if inter_edges:
            for k in sorted(inter_edges):
                eA = dict_A[k]
                # If both attribution dicts provided, show both scores
                if (attribution_scores_A is not None) and (
                    attribution_scores_B is not None
                ):
                    score_A = attribution_scores_A[eA.dest.module_name][
                        eA.patch_idx
                    ].item()
                    score_B = attribution_scores_B[eA.dest.module_name][
                        eA.patch_idx
                    ].item()
                    print(
                        f"  • {eA.src.name} → {eA.dest.name}  attr_score(A)= {score_A:.2f}, attr_score(B)= {score_B:.2f}"
                    )
                elif attribution_scores_A is not None:
                    score_A = attribution_scores_A[eA.dest.module_name][
                        eA.patch_idx
                    ].item()
                    print(
                        f"  • {eA.src.name} → {eA.dest.name}  attr_score(A)= {score_A:.2f}"
                    )
                elif attribution_scores_B is not None:
                    score_B = attribution_scores_B[eA.dest.module_name][
                        eA.patch_idx
                    ].item()
                    print(
                        f"  • {eA.src.name} → {eA.dest.name}  attr_score(B)= {score_B:.2f}"
                    )
                else:
                    print(f"  • {eA.src.name} → {eA.dest.name}")
        else:
            print("  (none)")

        print("\n=== Nodes only in A \\ B ===")
        only_A_nodes = nodes_A - nodes_B
        if only_A_nodes:
            for n in sorted(only_A_nodes):
                print(f"  • {n}")
        else:
            print("  (none)")

        print("\n=== Nodes only in B \\ A ===")
        only_B_nodes = nodes_B - nodes_A
        if only_B_nodes:
            for n in sorted(only_B_nodes):
                print(f"  • {n}")
        else:
            print("  (none)")

        print("\n=== Nodes in A ∩ B ===")
        if inter_nodes:
            for n in sorted(inter_nodes):
                print(f"  • {n}")
        else:
            print("  (none)")

    return {
        "edge_intersection": edge_intersection,
        "edge_union": edge_union,
        "edge_jaccard": edge_jaccard,
        "node_intersection": node_intersection,
        "node_union": node_union,
        "node_jaccard": node_jaccard,
    }


def save_comparison_plot(
    path_A, path_B, remaining_edges_A, remaining_edges_B, output_path, dpi=600
):
    data = {
        "Circuit A": {
            "edges": remaining_edges_A,
            "path": path_A,
        },
        "Circuit B": {
            "edges": remaining_edges_B,
            "path": path_B,
        },
    }
    base_img = Image.open(data["Circuit A"]["path"])
    ablated_img = Image.open(data["Circuit B"]["path"])

    # Resize with high-quality interpolation
    target_size = (1300, 1300)
    base_img_resized = base_img.resize(target_size, Image.Resampling.LANCZOS)
    ablated_img_resized = ablated_img.resize(target_size, Image.Resampling.LANCZOS)

    # Create larger figure with higher DPI
    plt.rcParams["figure.dpi"] = 300
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot images with captions
    ax1.imshow(base_img_resized)
    ax1.set_xlabel(f'Circuit A\n({data["Circuit A"]["edges"]} edges)')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(ablated_img_resized)
    ax2.set_xlabel(f'Circuit B\n({data["Circuit B"]["edges"]} edges)')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Save based on file extension
    plt.tight_layout()
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".svg":
        plt.savefig(output_path, format="svg", bbox_inches="tight")
    else:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_circuit_overlap_vs_accuracy(
    epochs: List[int],
    accuracies: List[float],
    circuit_overlap_logs: List[Dict],
    save_dir: str = None,
) -> None:
    """
    Plot bar charts showing node and edge overlaps vs task accuracy.

    Args:
        epochs: List of epoch numbers.
        accuracies: List of task accuracies after each epoch.
        circuit_overlap_logs: List of dictionaries with keys:
            - "node_intersection"
            - "node_union"
            - "edge_intersection"
            - "edge_union"
        save_dir: Directory where to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(
        {
            "Epoch": epochs,
            "Accuracy": accuracies,
            "Edge Intersection": [
                log["edge_intersection"] for log in circuit_overlap_logs
            ],
            "Edge Union": [log["edge_union"] for log in circuit_overlap_logs],
            "Node Intersection": [
                log["node_intersection"] for log in circuit_overlap_logs
            ],
            "Node Union": [log["node_union"] for log in circuit_overlap_logs],
        }
    )

    # Plot 1: Node Overlap vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(
        df["Accuracy"] - 0.01,
        df["Node Union"],
        width=0.02,
        label="All Nodes",
        alpha=0.5,
    )
    plt.bar(
        df["Accuracy"],
        df["Node Intersection"],
        width=0.02,
        label="Intersecting Nodes",
        alpha=0.8,
    )
    plt.xlabel("Accuracy")
    plt.ylabel("Number of Nodes")
    plt.title("Node Overlap vs Task Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "node_overlap_vs_accuracy.png"))
    plt.close()

    print(f"saving to {os.path.join(save_dir, 'node_overlap_vs_accuracy.png')}")

    # Plot 2: Edge Overlap vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(
        df["Accuracy"] - 0.01,
        df["Edge Union"],
        width=0.02,
        label="All Edges",
        alpha=0.5,
    )
    plt.bar(
        df["Accuracy"],
        df["Edge Intersection"],
        width=0.02,
        label="Intersecting Edges",
        alpha=0.8,
    )
    plt.xlabel("Accuracy")
    plt.ylabel("Number of Edges")
    plt.title("Edge Overlap vs Task Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "edge_overlap_vs_accuracy.png"))
    plt.close()


def plot_circuit_ablation(
    epochs: List[int],
    circuit_ablation_logs: List[Dict],
    save_dir: str = None,
) -> None:
    """


    Args:
        epochs: List of epoch numbers.
        accuracies: List of task accuracies after each epoch.
        circuit_ablation_logs: List of dictionaries with keys:
            - "avg_logit_diff_after_patching_A"
            - "avg_logit_diff_after_patching_B"
            - "proportion_correct_after_patching_A"
            - "proportion_correct_after_patching_B"
        save_dir: Directory where to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    # print(len(epochs), len(circuit_ablation_logs))
    
    def to_scalar(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item()
        return float(x)

    df = pd.DataFrame({
        "Epoch": epochs,
        "Average Logit Diff after patching A": [to_scalar(log["avg_logit_diff_after_patching_A"]) for log in circuit_ablation_logs],
        "Average Logit Diff after patching B": [to_scalar(log["avg_logit_diff_after_patching_B"]) for log in circuit_ablation_logs],
        "Proportion Correct after patching A": [to_scalar(log["proportion_correct_after_patching_A"]) for log in circuit_ablation_logs],
        "Proportion Correct after patching B": [to_scalar(log["proportion_correct_after_patching_B"]) for log in circuit_ablation_logs],
    })

    # Plot: Logit Difference
    plt.figure()
    plt.plot(df["Epoch"], df["Average Logit Diff after patching A"], label="Average Logit Diff after patching A", marker="o")
    plt.plot(df["Epoch"], df["Average Logit Diff after patching B"], label="Average Logit Diff after patching B", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Logit Difference")
    plt.title("Logit Difference after Circuit Ablation")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "logit_diff_vs_epoch.png"))
    plt.close()

    # Plot: Accuracy
    plt.figure()
    plt.plot(df["Epoch"], df["Proportion Correct after patching A"], label="Proportion Correct after patching A", marker="o")
    plt.plot(df["Epoch"], df["Proportion Correct after patching B"], label="Proportion Correct after patching B", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Proportion Correct")
    plt.title("Accuracy after Circuit Ablation")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.png"))
    plt.close()