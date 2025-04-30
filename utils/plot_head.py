from typing import List, Tuple, Optional, Union
from typing_extensions import Literal
import copy
import plotly.express as px
import plotly.graph_objects as go
from functools import partial
import numpy as np
import pandas as pd
import re

import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import to_numpy


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        # if isinstance(tensor[0])
        tensor = list(map(to_numpy, tensor))
        array = np.array(tensor)
        if array.dtype != np.dtype("O"):
            return array
        else:
            return to_numpy_ragged_2d(tensor)
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.detach().cpu().numpy()
    elif type(tensor) in [int, float, bool, str]:
        return np.array(tensor)
    elif isinstance(tensor, pd.Series):
        return tensor.values
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")
    
def to_numpy_ragged_2d(lists):
    # Assumes input is a ragged list (of lists, tensors or arrays). Further assumes it's 2D
    lists = list(map(to_numpy, lists))
    a = len(lists)
    b = max(map(len, lists))
    base_array = np.ones((a, b))
    base_array.fill(np.NINF)
    for i in range(a):
        base_array[i, : len(lists[i])] = lists[i]
    return base_array

# Defining Kwargs
DEFAULT_KWARGS = dict(
    xaxis="x",  # Good
    yaxis="y",  # Good
    range_x=None,  # Good
    range_y=None,  # Good
    animation_name="snapshot",  # Good
    
    color_name="Viridis",  # Good
    color="Viridis",
    log_x=False,  # Good
    log_y=False,  # Good
    toggle_x=False,  # Good
    toggle_y=False,  # Good
    legend=True,  # Good
    hover=None,  # Good
    hover_name="data",  # GOod
    return_fig=False,  # Good
    animation_index=None,  # Good
    line_labels=None,  # Good
    markers=False,  # Good
    frame_rate=None,  # Good
    facet_labels=None,
    facet_name="facet",
    include_diag=False,
    debug=False,
    transition="none",  # If "none" then turns off animation transitions, it just jumps between frames
    animation_maxrange_x=True,  # Figure out the maximal range if animation across all frames and fix
    animation_maxrange_y=True,  # Figure out the maximal range if animation across all frames and fix
)

def split_kwargs(kwargs):
    custom = dict(DEFAULT_KWARGS)
    plotly = {}
    for k, v in kwargs.items():
        if k in custom:
            custom[k] = v
        else:
            plotly[k] = v
    return custom, plotly

## Global Helpers
def update_data(data, custom_kwargs, index):
    if custom_kwargs["hover"] is not None and isinstance(data, go.Heatmap):
        # Assumption -
        hover = custom_kwargs["hover"]
        hover_name = custom_kwargs["hover_name"]
        hover = to_numpy(hover)
        data.customdata = hover
        update_hovertemplate(data, f"{hover_name}=%{{customdata}}")
    if custom_kwargs["markers"]:
        data["mode"] = "lines+markers"
    if custom_kwargs["line_labels"] is not None:
        data["name"] = custom_kwargs["line_labels"][index]
        data["hovertemplate"] = re.sub(
            f"={index}", f"={data['name']}", data["hovertemplate"]
        )
    return


def update_data_list(data_list, custom_kwargs):
    for c, data in enumerate(data_list):
        update_data(data, custom_kwargs, c)
    return


def update_frame(frame, custom_kwargs, frame_index):
    # if custom_kwargs['animation_index'] is not None:
    #     frame['name'] = custom_kwargs['animation_index'][frame_index]
    update_data_list(frame["data"], custom_kwargs)
    return

def update_play_button(button, custom_kwargs):
    if custom_kwargs["transition"] == "none":
        button.args[1]["transition"]["duration"] = 0
    else:
        button.args[1]["transition"]["easing"] = custom_kwargs["transition"]
    if custom_kwargs["frame_rate"] is not None:
        button.args[1]["frame"]["duration"] = custom_kwargs["frame_rate"]


def update_hovertemplate(data, string):
    if data.hovertemplate is not None:
        data.hovertemplate = (
            data.hovertemplate[:-15] + "<br>" + string + "<extra></extra>"
        )


def add_button(layout, button, pos=None):
    if pos is None:
        num_prev_buttons = len(layout.updatemenus)
        button["y"] = 1 - num_prev_buttons * 0.15
    else:
        button["y"] = pos
    if "x" not in button:
        button["x"] = -0.1
    layout.updatemenus = layout.updatemenus + (button,)


def add_axis_toggle(layout, axis, pos=None):
    assert axis in "xy", f"Invalid axis: {axis}"
    is_already_log = layout[f"{axis}axis"].type == "log"
    toggle_axis = dict(
        type="buttons",
        active=0 if is_already_log else -1,
        buttons=[
            dict(
                label=f"Log {axis}-axis",
                method="relayout",
                args=[{f"{axis}axis.type": "log"}],
                args2=[{f"{axis}axis.type": "linear"}],
            )
        ],
    )
    add_button(layout, toggle_axis, pos=pos)

def update_layout(layout, custom_kwargs, is_animation):
    if custom_kwargs["debug"]:
        print(layout, is_animation)
    layout.xaxis.title.text = custom_kwargs["xaxis"]
    layout.yaxis.title.text = custom_kwargs["yaxis"]
    if custom_kwargs["log_x"]:
        layout.xaxis.type = "log"
        if custom_kwargs["range_x"] is not None:
            range_x_0, range_x_1 = custom_kwargs["range_x"]
            layout.xaxis.range = (np.log10(range_x_0), np.log10(range_x_1))
    else:
        if custom_kwargs["range_x"] is not None:
            layout.xaxis.range = custom_kwargs["range_x"]
    if custom_kwargs["log_y"]:
        layout.yaxis.type = "log"
        if custom_kwargs["range_y"] is not None:
            range_y_0, range_y_1 = custom_kwargs["range_y"]
            layout.yaxis.range = (np.log10(range_y_0), np.log10(range_y_1))
    else:
        if custom_kwargs["range_y"] is not None:
            layout.yaxis.range = custom_kwargs["range_y"]
    if custom_kwargs["toggle_x"]:
        add_axis_toggle(layout, "x")
    if custom_kwargs["toggle_y"]:
        add_axis_toggle(layout, "y")
    if not custom_kwargs["legend"]:
        layout.showlegend = False
    if custom_kwargs["facet_labels"]:
        for i, label in enumerate(custom_kwargs["facet_labels"]):
            layout.annotations[i]["text"] = label
            if i > 0:
                layout[f"xaxis{i+1}"].title = layout["xaxis"].title

    if is_animation:
        for updatemenu in layout.updatemenus:
            if "buttons" in updatemenu:
                for button in updatemenu["buttons"]:
                    if button.label == "&#9654;":
                        update_play_button(button, custom_kwargs)
        layout.sliders[0].currentvalue.prefix = custom_kwargs["animation_name"] + "="
        if custom_kwargs["animation_index"] is not None:
            steps = layout.sliders[0].steps
            for c, step in enumerate(steps):
                step.label = custom_kwargs["animation_index"][c]


def update_fig(fig, custom_kwargs, inplace=True):
    if custom_kwargs["debug"]:
        print(fig.frames == tuple())
    if not inplace:
        fig = copy.deepcopy(fig)
    update_data_list(fig["data"], custom_kwargs)
    is_animation = "frames" in fig and fig.frames != tuple()
    if is_animation:
        for frame_index, frame in enumerate(fig["frames"]):
            update_frame(frame, custom_kwargs, frame_index)
    update_layout(fig.layout, custom_kwargs, is_animation)
    return fig


def imshow_base(array, **kwargs):
    array = to_numpy(array)
    custom_kwargs, plotly_kwargs = split_kwargs(kwargs)
    array = to_numpy(array)
    fig = px.imshow(array, **plotly_kwargs)
    update_fig(fig, custom_kwargs)
    if custom_kwargs["return_fig"]:
        return fig
    else:
        fig.show()


imshow = partial(
    imshow_base,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    aspect="auto",
)
imshow_pos = partial(imshow_base, color_continuous_scale="Blues", aspect="auto")

legend_in_plot_dict = dict(
    xanchor="right",
    x=0.95,
    title="",
    orientation="h",
    y=1.0,
    yanchor="top",
    bgcolor="rgba(255, 255, 255, 0.3)",
)

# Taken from https://github.com/callummcdougall/SERI-MATS-2023-Streamlit-pages/blob/main/transformer_lens/cautils/plotly_utils.py#L17
# Define a set of arguments which are passed to fig.update_layout (rather than just being included in e.g. px.imshow)
UPDATE_LAYOUT_SET = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat",
    "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_type", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor",
    "showlegend", "xaxis_tickmode", "yaxis_tickmode", "xaxis_tickangle", "yaxis_tickangle", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "font",
    "modebar_add", "legend_traceorder", "autosize", "coloraxis_colorbar_tickformat", "font_family", "font_size",
}
# Gives options to draw on plots, and remove plotly logo
CONFIG = {'displaylogo': False}
CONFIG_STATIC = {'displaylogo': False, 'staticPlot': True}
MODEBAR_ADD = ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']

def hist(tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in UPDATE_LAYOUT_SET}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in UPDATE_LAYOUT_SET}
    draw = kwargs_pre.pop("draw", True)
    static = kwargs_pre.pop("static", False)
    return_fig = kwargs_pre.pop("return_fig", False)
    if isinstance(tensor, list):
        if isinstance(tensor[0], (torch.Tensor, np.ndarray)): arr = [to_numpy(tn) for tn in tensor]
        elif isinstance(tensor[0], list): arr = [np.array(tn) for tn in tensor]
        else: arr = np.array(tensor)
    else:
        arr = to_numpy(tensor)
    if "modebar_add" not in kwargs_post:
        kwargs_post["modebar_add"] = ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
    add_mean_line = kwargs_pre.pop("add_mean_line", False)
    names = kwargs_pre.pop("names", None)
    if "barmode" not in kwargs_post:
        kwargs_post["barmode"] = "overlay"
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"
    if "autosize" not in kwargs_post:
        kwargs_post["autosize"] = False
    # print(kwargs_pre, "and", kwargs_post)
    fig = px.histogram(x=arr, **kwargs_pre)
    fig.update_layout(**kwargs_post)
    if add_mean_line:
        if arr.ndim == 1:
            fig.add_vline(x=arr.mean(), line_width=3, line_dash="dash", line_color="black", annotation_text=f"Mean = {arr.mean():.3f}", annotation_position="top")
        elif arr.ndim == 2:
            for i in range(arr.shape[0]):
                fig.add_vline(x=arr[i].mean(), line_width=3, line_dash="dash", line_color="black", annotation_text=f"Mean = {arr.mean():.3f}", annotation_position="top")
    if names is not None:
        for i in range(len(fig.data)):
            fig.data[i]["name"] = names[i // 2 if "marginal" in kwargs_pre else i]
    if draw: 
        fig.update_layout(modebar_add=MODEBAR_ADD)
    else:
        fig.update_layout(modebar_add=[])
    if return_fig:
        return fig
    else:
        fig.show(renderer=renderer, config=CONFIG_STATIC if static else CONFIG)


def show_attention_patterns(
    model: HookedTransformer,
    heads: List[Tuple[int, int]],
    prompts: List[str],
    precomputed_cache: Optional[ActivationCache] = None,
    mode: Union[Literal["val", "pattern", "scores"]] = "val",
    title_suffix: Optional[str] = "",
    return_fig: bool = False,
    return_mtx: bool = False,
):
    """
    Visualizes the different types of attention for the specified heads in the model.

    Args:

    model (torch.nn.Module): Model to visualize.
    heads (List[Tuple[int, int]]): List of tuples specifying the layer and head indices to visualize.
    prompts (List[str]): List of prompt sequences. The first sequence is used for visualization.
    precomputed_cache (Dict[str, torch.Tensor], optional): Precomputed activations cache.
    mode (str): Visualization mode ('pattern', 'val', 'scores').
    title_suffix (str): Suffix to append to the plot title.
    return_fig (bool): Whether to return the plotly figure.
    return_mtx (bool): Whether to return the attention matrices.

    Returns:
    - If return_fig=True and return_mtx=False, returns the plotly figure.
    - If return_fig=False and return_mtx=True, returns the attention matrices.
    - If return_fig=False and return_mtx=False, displays the attention patterns.

    Info:
    - 'scores': Visualizes the attention scores pre-softmax, 
    - 'pattern': Visualizes the attention patterns post-softmax, or attention probabilities,
    - 'val-weighted': Visualizes the value-weighted attention patterns,
    - 'ov': Visualizes the OV circuit

    Note: 
    ! More about the types of activations on:  https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.ActivationCache.html
    """
    assert mode in [
        "pattern",
        "scores",
        "val-weighted",
        "ov",
    ] 
    assert len(heads) == 1 or not (return_fig or return_mtx)

    for (layer, head) in heads:
        cache = {}
        good_names = []
        good_names.append(f"blocks.{layer}.attn.hook_v")  # (batch, pos, head_index, d_head)
        good_names.append(f"blocks.{layer}.attn.hook_pattern")  # (batch, head_index, query_pos, key_pos)
        good_names.append(f"blocks.{layer}.attn.hook_attn_scores")  # (batch, head_index, query_pos, key_pos)
        good_names.append(f"blocks.{layer}.attn.hook_z")  #  (batch, pos, head_index, d_head)
        good_names.append(f"blocks.{layer}.hook_attn_out")   # (batch, )
        
        # else:
        if precomputed_cache is None:
            cache = {}
            def hook_fn(activation: torch.Tensor, hook: HookPoint, name: str = "activation"):
                '''Stores activations in hook context.'''
                cache[name] = activation
                return activation

            fwd_hooks = [
                (good_names[0], partial(hook_fn, name=good_names[0])), 
                (good_names[1], partial(hook_fn, name=good_names[1])), 
                (good_names[2], partial(hook_fn, name=good_names[2])), 
                (good_names[3], partial(hook_fn, name=good_names[3])), 
                (good_names[4], partial(hook_fn, name=good_names[4])), 
            ]
            model.run_with_hooks(prompts, fwd_hooks=fwd_hooks)
        else:
            cache = precomputed_cache
         
        attn_results = torch.zeros(
            size=(len(prompts), len(prompts[0]), len(prompts[0]))
        )
        attn_results += -20

        # TODO: BOS token will be a problem for diagonal heads, 
        toks = model.to_tokens(prompts)
        # toks = utils.get_tokens_with_bos_removed(model.tokenizer, toks)
        current_length = len(toks)
        words = model.to_str_tokens(prompts)
        # print(f"Before trying to remove bos: {words}")
        # # if model.tokenizer.pad_token_id in toks: 
        
        # words = words[1:]
        # print(f"After trying to remove bos: {words}")
        attn_pattern = cache[good_names[1]].detach().cpu()[:, head, :, :].squeeze(0)
        attn_scores = cache[good_names[2]].detach().cpu()[:, head, :, :].squeeze(0)
        if mode == "val-weighted":
            if getattr(model.cfg, "ungroup_grouped_query_attention", True):
                # n_heads == n_key_value_heads
                vals = cache[good_names[0]].detach().cpu()[0, :, head, :].norm(p=2,dim=-1)
                cont = attn_pattern * vals.unsqueeze(0)

            else:
                # use GQA kv_head grouping
                kv_group_size = model.cfg.n_heads // model.cfg.n_key_value_heads
                kv_head = head // kv_group_size
                vals = cache[good_names[0]].detach().cpu()[0, :, kv_head, :].norm(p=2,dim=-1)
                cont = attn_pattern * vals.unsqueeze(0)

        labels={"y": "Queries", "x": "Keys"}
        # TODO: Plotting OV works only with the effective circuit
        if mode == "ov": 
            # print(f"Shape of value: {cache[good_names[0]].shape}") # value
            # print(f"Shape of attn_out: {cache[good_names[4]].shape}") # attn_out
            # v_act = cache[good_names[0]].detach().cpu().squeeze(0)  # -> shape [seq, n_kv_heads, d_head]
            # cont = cache[good_names[4]].detach().cpu().squeeze(0)  # -> shape [n_kv_heads, seq, d_head]
            # if getattr(model.cfg, "n_key_value_heads"):
            #     kv_group_size = model.cfg.n_heads // model.cfg.n_key_value_heads
            #     kv_head = head // kv_group_size

            #     # Extract single query head
            #     v_act = v_act[:, head, :]  # (seq, d_head)
            #     # Extract corresponding kv head
            #     cont_head = cont[kv_head, :]  # (seq, d_head)
            # else: 
            #     kv_head = head
            #     cont_head = cont[kv_head, :]
            
            # cont = v_act @ cont_head.T  # shapes [seq, d_head] x [d_head, seq] → [seq, seq]
            
            W_V_tmp, W_O_tmp = model.W_V[layer, head, :], model.W_O[layer, head]
            tmp = model.W_E @ W_V_tmp @ W_O_tmp @ model.W_U

            input_tokens = toks[0]  
            seq_len = input_tokens.shape[0]
            cont = torch.zeros((seq_len, seq_len))

            for i in range(seq_len):
                for j in range(seq_len):
                    cont[i, j] = tmp[input_tokens[i], input_tokens[j]]
            cont = cont.detach().cpu()

            labels={"y": "Output Token", "x": "Source Token"},
        
        fig = px.imshow(
            attn_pattern if mode == "pattern" else attn_scores if mode == "scores" else cont,
            title=f"{layer}.{head} Attention" + title_suffix,
            color_continuous_midpoint=0,
            color_continuous_scale="RdBu",
            labels=labels,
            height=600,
        )

        fig.update_layout(
            xaxis={
            "side": "top",
            "ticktext": words,
            "tickvals": list(range(len(words))),
            "tickfont": dict(size=15),
            },
            yaxis={
            "ticktext": words,
            "tickvals": list(range(len(words))),
            "tickfont": dict(size=15),
            },
            width=800, 
            height=650,
        )
        if return_fig and not return_mtx:
            return fig
        elif return_mtx and not return_fig:
            if mode == "val-weighted":
                return cont
            elif mode == "pattern":
                attn_results[:, :current_length, :current_length] = (
                    attn_pattern[:current_length, :current_length].clone().cpu()
                )
            elif mode == "scores":
                attn_results[:, :current_length, :current_length] = (
                    attn_scores[:current_length, :current_length].clone().cpu()
                )
        else:
            fig.show()

        if return_fig and not return_mtx:
            return fig
        elif return_mtx and not return_fig:
            return attn_results