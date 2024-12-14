import torch
import e3nn
from torch.utils.cpp_extension import load
import json
import matplotlib.pyplot as plt
import itertools
import os,sys
import cuequivariance as cue
import cuequivariance_torch as cuet
import random

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0"
sptp_bwd = load(name='sptp_linear_bwd', sources=['/home2/lsy/mdsim/fused_e3nn/fused_e3nn_kernel/sptp_linear_bwd.cpp', 
                                  '/home2/lsy/mdsim/fused_e3nn/fused_e3nn_kernel/bwd_sptp_linear.cu',
                                  '/home2/lsy/mdsim/fused_e3nn/fused_e3nn_kernel/bwd_sptp_linear_shared.cu',
                                  '/home2/lsy/mdsim/fused_e3nn/fused_e3nn_kernel/fwd_sptp_linear_v2.cu',
                                  '/home2/lsy/mdsim/fused_e3nn/fused_e3nn_kernel/fwd_sptp_linear_v2_shared.cu',
                                  ], 
                                  extra_cuda_cflags=["-lineinfo"], verbose=True)


def mul_Irreps(mul, i_in):
    dd = []
    for ori_mul, ir in i_in:
        dd.append((ori_mul*mul, (ir.l, ir.p)))
    return e3nn.o3.Irreps(dd)
def compare(a, b):
    isclose = torch.isclose(a, b)
    diff_pos = torch.argwhere(isclose == False)
    anything_bad = False
    for pos in diff_pos:
        pos_t = [x for x in pos]
        if(abs(a[pos_t] - b[pos_t]) > 1e-6):
            anything_bad = True
            print(pos)
            print(a[pos_t] - b[pos_t] )
    if(not anything_bad):
        print("All Good")
            
IR_IN1_IDX = 0
IR_IN2_IDX = 1
IR_OUT_IDX = 2
INST_IDX = 3
WARPSIZE = 32

def load_nequip_config(h, l_max, layer_idx):
    filename = f"/home2/lsy/mdsim/nequip/benchmark_config/4_{h}_{l_max}_p_sc.txt"
    with open(filename, "r") as f:
        f_in = f.read().split("\n")

    per_layer_dict = dict()
    for l_idx, d in enumerate(f_in):
        if(d == "") : continue
        dd = json.loads(d)
        per_layer_dict[l_idx] = dd
    tp_list = per_layer_dict[layer_idx]["tp"]
    i_in1 = e3nn.o3.Irreps(tp_list[IR_IN1_IDX])
    i_in2 = e3nn.o3.Irreps(tp_list[IR_IN2_IDX])
    i_out = e3nn.o3.Irreps(tp_list[IR_OUT_IDX])
    inst_tuple = [tuple(x) for x in tp_list[INST_IDX]]

    return i_in1, i_in2, i_out, inst_tuple

def load_nequip_config_e3nn_cueq(h, l_max, layer_idx):
    filename = f"/home2/lsy/mdsim/nequip/benchmark_config/4_{h}_{l_max}_p_sc.txt"
    with open(filename, "r") as f:
        f_in = f.read().split("\n")

    per_layer_dict = dict()
    for l_idx, d in enumerate(f_in):
        if(d == "") : continue
        dd = json.loads(d)
        per_layer_dict[l_idx] = dd
    tp_list = per_layer_dict[layer_idx]["tp"]

    ei_in1 = e3nn.o3.Irreps(tp_list[IR_IN1_IDX])
    ei_in2 = e3nn.o3.Irreps(tp_list[IR_IN2_IDX])
    ei_out = e3nn.o3.Irreps(tp_list[IR_OUT_IDX])
    inst_tuple = [tuple(x) for x in tp_list[INST_IDX]]


    # changing mul for each ir.l
    new_in1_list = []
    new_out_list = []
    changed_idx = [[],[]]
    # mul_list = {}
    mul_list = {0:128, 1:64}

    for idx, (mul,ir) in enumerate(ei_in1):
        if (ir.l in mul_list):
            new_in1_list.append((mul_list[ir.l], ir))
            for inst in inst_tuple:
                if(idx == inst[0]):
                    changed_idx[0].append(inst[2])
                    changed_idx[1].append(mul_list[ir.l])
        else:
            new_in1_list.append((mul, ir))

    for idx, (mul,ir) in enumerate(ei_out):
        if (idx in changed_idx[0]):
            new_out_list.append((changed_idx[1][changed_idx[0].index(idx)], ir))
        else:
            new_out_list.append((mul, ir))

    ei_in1 = e3nn.o3.Irreps(new_in1_list)
    ei_out = e3nn.o3.Irreps(new_out_list)

    ci_in1 = cue.Irreps("O3", str(ei_in1))
    ci_in2 = cue.Irreps("O3", tp_list[IR_IN2_IDX])
    ci_out = cue.Irreps("O3", str(ei_out))


    return [ei_in1,ei_in2,ei_out,inst_tuple] , [ci_in1,ci_in2,ci_out,inst_tuple]

def main():
    h = int(sys.argv[1])
    l_max = int(sys.argv[2])
    layer_idx = int(sys.argv[3])
    tp_type = sys.argv[4]
    batch_size = int(sys.argv[5])
    use_compare = sys.argv[6] == "True"
    block_batch_cnt = int(sys.argv[7])

    e3nn_config, cueq_config = load_nequip_config_e3nn_cueq(h,l_max,layer_idx)
    i_in1, i_in2, i_out, inst_tuple = e3nn_config

    uvuv_tp = e3nn.o3.FullTensorProduct(i_in1,i_in2, filter_ir_out=i_out)
    uvuv_i_out = uvuv_tp.irreps_out
    ## weight rearrangment
    # split_size = []
    # reshape_size = []
    # for inst in uvuv_tp.instructions:
    #     split_size.append(uvuv_i_out[inst.i_out].dim)
    #     reshape_size.append([inst.path_shape[0],inst.path_shape[1],uvuv_i_out[inst.i_out][1].dim])
    # weight_mul = e3nn.o3.experimental.FullTensorProduct_uvu_weight_only(i_in1, i_in2, split_size, reshape_size, filter_ir_out=i_out, irrep_normalization=None, regroup_output=False)


    # # uvu
    tp = e3nn.o3.TensorProduct(i_in1,i_in2,i_out,inst_tuple,shared_weights=False, internal_weights=False) # path_normalization="none", normalization="none"
    tp = tp.to(device="cuda")

    cuet_tp = cuet.ChannelWiseTensorProduct(*cueq_config[:-1], shared_weights=False,internal_weights=False,device="cuda", layout="ir_mul")
    # irreps3 = []
    # for (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
    #     enumerate(cueq_config[0]), enumerate(cueq_config[1])
    # ):
    #     for ir3 in ir1 * ir2:
    #         if ir3 not in cueq_config[2]:
    #             continue

    #         # for loop over the different solutions of the Clebsch-Gordan decomposition
    #         for cg in cue.clebsch_gordan(ir1, ir2, ir3):
    #             # d.add_path(None, i1, i2, None, c=cg, dims={"u": mul1, "v": mul2})

    #             irreps3.append((mul1 * mul2, ir3))

    # irreps3 = cue.Irreps("O3", irreps3)
    # _,_, inv  = irreps3.sort()
    # ceq_weight_slice = []
    # current = 0
    # for mul,ir in irreps3:
    #     ceq_weight_slice.append(slice(current,current+mul,None))
    #     current+=mul

    # full tp -> linear
    unique_cg = []
    unique_cg_mat = {}
    cg_dummy = torch.zeros(i_in1.dim, i_in2.dim, uvuv_i_out.dim)
    for inst in uvuv_tp.instructions:
        i = inst.i_in1
        j = inst.i_in2
        k = inst.i_out

        mul_in1, ir_in1 = i_in1[i]
        mul_in2, ir_in2 = i_in2[j]
        mul_out, ir_out = uvuv_i_out[k]

        cg = e3nn.o3.wigner_3j(ir_in1.l, ir_in2.l, ir_out.l)
        unique_cg += list(cg.unique())

        partial_mat_cg = torch.zeros(i_in1[i].dim, i_in2[j].dim, uvuv_i_out[k].dim)
        # print(cg)
        unique_cg_mat[f"{ir_in1.l}_{ir_in2.l}_{ir_out.l}"] = cg
        
        ## uvuv
        for u,v in itertools.product(range(mul_in1), range(mul_in2)):
            partial_mat_cg [u*ir_in1.dim:(u+1)*ir_in1.dim,
            v*ir_in2.dim:(v+1)*ir_in2.dim,
            (u*mul_in2+v)*ir_out.dim:(u*mul_in2+v+1)*ir_out.dim] = cg 

        cg_dummy[i_in1.slices()[i], i_in2.slices()[j], uvuv_i_out.slices()[k]] = partial_mat_cg

    unique_cg_val = list(set([x.item() for x in unique_cg]))
    tp_inst_outorder = sorted(tp.instructions, key=lambda x : x.i_out)
    
    # already duplicate as num_path > num unique cg matrix
    per_path_fiber_start = [0]
    per_path_fiber_array = []

    for inst in tp_inst_outorder:
        path_cg = unique_cg_mat[f"{i_in1[inst.i_in1][1].l}_{i_in2[inst.i_in2][1].l}_{i_out[inst.i_out][1].l}"]
        for i,j,k in path_cg.nonzero():
            cg_idx = unique_cg_val.index(path_cg[i,j,k])
            per_path_fiber_array.append([i.item(),j.item(),k.item(),cg_idx])
        per_path_fiber_start.append(len(per_path_fiber_array))

    per_in1_ir_pathinfo = {}
    for inst in tp_inst_outorder:
        if(inst.i_in1 not in per_in1_ir_pathinfo):
            per_in1_ir_pathinfo[inst.i_in1] = []
        per_in1_ir_pathinfo[inst.i_in1].append([inst.i_out, inst.i_in2, inst.path_weight])

    weight_uv_pair = []
    weight_uv_pair_sorted_chunk = []
    out_order = []
    current = 0

    for inst in tp.instructions:
        weight_uv_pair.append((i_in1[inst.i_in1][0], i_in2[inst.i_in2][0] ))
        out_order.append(inst.i_out)
    for u,v in weight_uv_pair:
        weight_uv_pair_sorted_chunk.append(slice(current,current+u*v))
        current+=u*v
    out2weight_order = torch.tensor(out_order).argsort()

    in1_idxing = [0]
    in1_ival = []
    in1_related_path_idx = [0]

    path_array1 = []
    path_array2 = []
    path_weight = []
    per_path_weight_pos = []

    per_upath_fiber_start = []
    per_upath_fiber_array = []

    in1_slices = i_in1.slices()
    in2_slices = i_in2.slices()
    out_slices = i_out.slices()

    for in1_ir_idx, (mul,ir) in enumerate(i_in1):
        assert (mul%WARPSIZE ==0)
        in1_idx_start = in1_slices[in1_ir_idx].start
        i_val = ir.dim
        
        if mul >= WARPSIZE:
            for i in range(mul//WARPSIZE):
                in1_idxing.append(in1_idx_start + WARPSIZE*i_val*(i+1))
                in1_ival.append(i_val)
                in1_related_path_idx.append(in1_related_path_idx[-1] + len(per_in1_ir_pathinfo[in1_ir_idx]))
                
                dummy_list = []
                dummy_list2 = []
                # Bug? TODO:
                for out_ir_idx, in2_ir_idx, pw in per_in1_ir_pathinfo[in1_ir_idx]:
                    # should be in order
                    fiber_start = per_path_fiber_start[out_ir_idx]
                    fiber_end = per_path_fiber_start[out_ir_idx+1]
                    
                    upath_fiber_start = len(per_upath_fiber_array)
                    upath_fiber_end = upath_fiber_start + fiber_end - fiber_start

                    per_upath_fiber_start.append([upath_fiber_start, upath_fiber_end])
                    # print(fiber_array_orignal[1:4])
                    # print(fiber_start,fiber_end)
                    # print(fiber_array_orignal[fiber_start:fiber_end])
                    
                    per_upath_fiber_array += per_path_fiber_array[fiber_start:fiber_end]

                    dummy_list.append([out_slices[out_ir_idx].start + WARPSIZE*i_out[out_ir_idx].ir.dim * i,
                                    out_slices[out_ir_idx].start + WARPSIZE*i_out[out_ir_idx].ir.dim * (i+1)
                                    ])
                    dummy_list2.append([i_out[out_ir_idx].ir.dim,
                                        in2_slices[in2_ir_idx].start,
                                        i_in2[in2_ir_idx].ir.dim,
                                        in2_slices[in2_ir_idx].stop])
                    path_weight.append(pw)
                    
                    # TODO:??
                    per_path_weight_pos.append(weight_uv_pair_sorted_chunk[out2weight_order[out_ir_idx]].start + WARPSIZE*i)


                path_array1.append(dummy_list)
                path_array2.append(dummy_list2)

    t_in1_idxing = torch.tensor(in1_idxing, dtype=torch.int32, device="cuda")
    t_in1_ival = torch.tensor(in1_ival, dtype=torch.int32, device="cuda")
    t_in1_related_path_idx = torch.tensor(in1_related_path_idx, dtype=torch.int32, device="cuda")

    t_path_array1 = torch.tensor(list(itertools.chain.from_iterable(path_array1)), dtype=torch.uint16, device="cuda")
    t_path_array2 = torch.tensor(list(itertools.chain.from_iterable(path_array2)), dtype=torch.uint8, device="cuda")
    t_path_weight = torch.tensor(path_weight, dtype=torch.float32, device="cuda")
    t_per_path_weight_pos = torch.tensor(per_path_weight_pos, dtype=torch.int32, device="cuda")

    t_per_upath_fiber_start = torch.tensor(per_upath_fiber_start, dtype=torch.uint16, device="cuda")
    t_per_upath_fiber_array = torch.tensor(per_upath_fiber_array, dtype=torch.uint8, device="cuda")

    t_unique_cg_val = torch.tensor(unique_cg_val, dtype=torch.float32, device="cuda")

    upath_cnt = len(in1_idxing)-1

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(10):
        in1 = torch.rand(batch_size, i_in1.dim, device="cuda", requires_grad=True)
        in2 = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True)
        weight = torch.rand(batch_size,tp.weight_numel, device="cuda", requires_grad=True)
        out_debug = torch.zeros((batch_size,i_out.dim), device="cuda")

        cin1_cuda = torch.rand(batch_size, i_in1.dim, device="cuda", requires_grad=True)
        cin2_cuda = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True)
        weight_cueq = torch.rand(batch_size,tp.weight_numel, device="cuda", requires_grad=True)

        # cueq weight 
        # weight_cueq = []
        # for inst_idx in inv:
        #     w_slice = ceq_weight_slice[inst_idx]
        #     weight_cueq.append(weight[:,w_slice])
        # weight_cueq = torch.cat(weight_cueq,dim=1)
        # weight.requires_grad = True
        # weight_cueq.requires_grad = True
        
        # tensor for ours bwd
        mem_dl_din1 = torch.zeros_like(in1)
        mem_dl_din2 = torch.zeros((batch_size, i_in2.dim * upath_cnt), device="cuda")
        mem_dl_dw = torch.zeros_like(weight)
        # mem_dl_do = torch.cat([x.grad.reshape(batch_size,-1) for x in grad_uvu.w_result_list],dim=1)
        # mem_dl_do = torch.ones((batch_size, i_out.dim), device="cuda")
        mem_debug = torch.ones((batch_size, i_out.dim), device="cuda") * -1

        print(i)
        # fwd path
        torch.cuda.nvtx.range_push(f"e3nn_fwd")
        out = tp(in1,in2,weight)
        out.retain_grad()
        y = out.sum()
        y.retain_grad()
        torch.cuda.nvtx.range_pop()

        # original e3nn
        torch.cuda.nvtx.range_push(f"e3nn_bwd")
        y.backward()
        torch.cuda.nvtx.range_push(f"e3nn_bwd_dummy")
        print(in1.grad[0])
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"sptp_fwd")
        sptp_bwd.sptp_linear_fwd_v2(in1,in2,weight, out_debug,
                            t_in1_idxing, t_in1_ival, t_in1_related_path_idx, t_path_array1,t_path_array2,t_per_upath_fiber_start, t_path_weight, t_per_path_weight_pos, t_per_upath_fiber_array,t_unique_cg_val, upath_cnt, block_batch_cnt,
                            )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"sptp_fwd_shared")
        sptp_bwd.sptp_linear_fwd_v2_shared(in1,in2,weight, out_debug,
                            t_in1_idxing, t_in1_ival, t_in1_related_path_idx, t_path_array1,t_path_array2,t_per_upath_fiber_start, t_path_weight, t_per_path_weight_pos, t_per_upath_fiber_array,t_unique_cg_val, upath_cnt, block_batch_cnt, l_max*2+1
                            )
        torch.cuda.nvtx.range_pop()
        print(out[0])
        print(out_debug[0])


        torch.cuda.nvtx.range_push(f"sptp_bwd_shared")
        # full sptp
        sptp_bwd.sptp_linear_bwd_v1_shared(in1,in2,weight, out.grad, mem_dl_din1, mem_dl_din2, mem_dl_dw, mem_debug,
                            t_in1_idxing, t_in1_ival, t_in1_related_path_idx, t_path_array1,t_path_array2,t_per_upath_fiber_start, t_path_weight, t_per_path_weight_pos, t_per_upath_fiber_array,t_unique_cg_val, upath_cnt, block_batch_cnt, l_max*2+1
                            )
        mem_dl_din2_summed = mem_dl_din2.reshape((batch_size, upath_cnt, i_in2.dim)).sum(dim=1)
        torch.cuda.nvtx.range_pop()
        print(mem_dl_din1[0])

        torch.cuda.nvtx.range_push(f"sptp_bwd")
        # full sptp
        sptp_bwd.sptp_linear_bwd_v1(in1,in2,weight, out.grad, mem_dl_din1, mem_dl_din2, mem_dl_dw, mem_debug,
                            t_in1_idxing, t_in1_ival , t_in1_related_path_idx, t_path_array1,t_path_array2,t_per_upath_fiber_start, t_path_weight, t_per_path_weight_pos, t_per_upath_fiber_array,t_unique_cg_val, upath_cnt, block_batch_cnt
                            )
        mem_dl_din2_summed = mem_dl_din2.reshape((batch_size, upath_cnt, i_in2.dim)).sum(dim=1)
        torch.cuda.nvtx.range_pop()
        print(mem_dl_din1[0])

        
        torch.cuda.nvtx.range_push(f"d")
        cuet_out = cuet_tp(cin1_cuda,cin2_cuda,weight_cueq)
        torch.cuda.nvtx.range_pop()
        y2 = cuet_out.sum()
        torch.cuda.nvtx.range_push(f"cueq_irmul_bwd")
        y2.backward()
        torch.cuda.nvtx.range_push(f"cueq_irmul_bwd_dummy")
        print(cin1_cuda.grad[0])
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()


        if(use_compare):
            # print(out_cuda)
            # print(e3nn_fc_out_rand)
            print("in1 grad")
            compare(in1.grad.cpu(), mem_dl_din1.cpu())
            print("weight grad")
            compare( weight.grad.cpu(), mem_dl_dw.cpu())
            print("in2 grad")
            compare( in2.grad.cpu(), mem_dl_din2_summed.cpu())

    torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    main()
