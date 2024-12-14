from copy import deepcopy
from typing import Callable, List, Optional, Tuple

from e3nn.o3 import Irreps

import sevenn._keys as KEY

from .convolution import CGAfterGatherConvolution, IrrepsConvolution, FusedE3nnConv
from .equivariant_gate import EquivariantGate
from .linear import IrrepsLinear


def NequIP_interaction_block(
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_out_tp: Irreps,
    irreps_out: Irreps,
    weight_nn_layers: List[int],
    conv_denominator: float,
    train_conv_denominator: bool,
    self_connection_pair: Tuple[Callable, Callable],
    act_scalar: Callable,
    act_gate: Callable,
    act_radial: Callable,
    bias_in_linear: bool,
    num_species: int,
    t: int,  # interaction layer index
    data_key_x: str = KEY.NODE_FEATURE,
    data_key_weight_input: str = KEY.EDGE_EMBEDDING,
    parallel: bool = False,
    multi_cutoff_list: Optional[List[int]] = None,
    **conv_kwargs,
):
    block = {}
    irreps_node_attr = Irreps(f'{num_species}x0e')
    sc_intro, sc_outro = self_connection_pair

    gate_layer = EquivariantGate(irreps_out, act_scalar, act_gate)
    irreps_for_gate_in = gate_layer.get_gate_irreps_in()

    block[f'{t}_self_connection_intro'] = sc_intro(
        irreps_x,
        irreps_operand=irreps_node_attr,
        irreps_out=irreps_for_gate_in,
    )

    block[f'{t}_self_interaction_1'] = IrrepsLinear(
        irreps_x,
        irreps_x,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    # convolution part, l>lmax is dropped as defined in irreps_out
    if multi_cutoff_list is None:
        data_key_filter = KEY.EDGE_ATTR
        data_key_edge_idx = KEY.EDGE_IDX
    else:
        data_key_weight_input = f'{data_key_weight_input}{multi_cutoff_list[t]}'
        data_key_filter = f'{KEY.EDGE_ATTR}{multi_cutoff_list[t]}'
        data_key_edge_idx = f'{KEY.EDGE_IDX}{multi_cutoff_list[t]}'

    conv_kwargs = deepcopy(conv_kwargs)
    use_cg_af_gat = conv_kwargs.pop('cg_af_gat', False)
    use_fused = conv_kwargs.pop('use_fused', False)
    if use_fused:
        block[f'{t}_convolution'] = FusedE3nnConv(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_out_tp,
            weight_layer_input_to_hidden=weight_nn_layers,
            weight_layer_act=act_radial,
            denominator=conv_denominator,
            train_denominator=train_conv_denominator,
            data_key_filter=data_key_filter,
            data_key_weight_input=data_key_weight_input,
            data_key_edge_idx=data_key_edge_idx,
            is_parallel=parallel,
            **conv_kwargs,
        )
    elif irreps_x.lmax == 0 or irreps_out.lmax == 0 or not use_cg_af_gat:
        block[f'{t}_convolution'] = IrrepsConvolution(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_out_tp,
            weight_layer_input_to_hidden=weight_nn_layers,
            weight_layer_act=act_radial,
            denominator=conv_denominator,
            train_denominator=train_conv_denominator,
            data_key_filter=data_key_filter,
            data_key_weight_input=data_key_weight_input,
            data_key_edge_idx=data_key_edge_idx,
            is_parallel=parallel,
            **conv_kwargs,
        )
    elif use_cg_af_gat:
        print(irreps_x, irreps_out)
        block[f'{t}_convolution'] = CGAfterGatherConvolution(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_out_tp,
            weight_layer_input_to_hidden=weight_nn_layers,
            weight_layer_act=act_radial,
            denominator=conv_denominator,
            train_denominator=train_conv_denominator,
            data_key_filter=data_key_filter,
            data_key_weight_input=data_key_weight_input,
            data_key_edge_idx=data_key_edge_idx,
            is_parallel=parallel,
            **conv_kwargs,
        )

    # irreps of x increase to gate_irreps_in
    block[f'{t}_self_interaction_2'] = IrrepsLinear(
        irreps_out_tp,
        irreps_for_gate_in,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    block[f'{t}_self_connection_outro'] = sc_outro()
    block[f'{t}_equivariant_gate'] = gate_layer

    return block
