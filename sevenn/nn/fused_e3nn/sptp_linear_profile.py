import torch
import e3nn
from torch.utils.cpp_extension import load
import json
import matplotlib.pyplot as plt
import itertools
import os,sys

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0"
sptp = load(name='sptp_linear', sources=['/home2/lsy/fused_e3nn/sptp_linear/sptp_linear.cpp', 
                                  '/home2/lsy/fused_e3nn/sptp_linear/sptp_linear.cu',
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
    for pos in diff_pos:
        pos_t = [x for x in pos]
        if(abs(a[pos_t] - b[pos_t]) > 1e-7):
            print(pos)
            print(a[pos_t] - b[pos_t] )
            
IR_IN1_IDX = 0
IR_IN2_IDX = 1
IR_OUT_IDX = 2
INST_IDX = 3

def load_nequip_config(h, l_max, layer_idx):
    filename = f"/home2/lsy/fused_e3nn/benchmark_config/4_{h}_{l_max}_p_sc.txt"
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

def to_cuda_list(*arg, input_dtype = torch.float32):
    return_list = []
    for item in arg:
        if(type(item) == torch.Tensor):
            return_list.append(item.to(device="cuda"))
        else:
            return_list.append(torch.tensor(item,device="cuda", dtype=input_dtype))
    return return_list

def to_cuda_dict(strname_list, *arg):
    return_dict = {}
    for item,name in zip(arg,strname_list):
        if(type(item) == torch.Tensor):
            return_dict[name] = item.to("cuda")
        else:
            return_dict[name] = torch.tensor(item,device="cuda")
    return return_dict

def cumsum_list(s, np1 = True):
    new_s = []
    current = 0
    for e in s:
        new_s.append(current)
        current += e
    if(np1):
        new_s.append(current)
    return new_s



def main():
    h = int(sys.argv[1])
    l_max = int(sys.argv[2])
    layer_idx = int(sys.argv[3])
    tp_type = sys.argv[4]
    batch_size = int(sys.argv[5])

    use_compare = sys.argv[6] == "True"

    i_in1, i_in2, i_out, inst_tuple = load_nequip_config(h,l_max,layer_idx)
    uvuv_tp = e3nn.o3.FullTensorProduct(i_in1,i_in2, filter_ir_out=i_out, path_normalization="none", normalization="none")
    uvuv_i_out = uvuv_tp.irreps_out
    ## weight rearrangment
    split_size = []
    reshape_size = []
    for inst in uvuv_tp.instructions:
        split_size.append(uvuv_i_out[inst.i_out].dim)
        reshape_size.append([inst.path_shape[0],inst.path_shape[1],uvuv_i_out[inst.i_out][1].dim])
    weight_mul = e3nn.o3.experimental.FullTensorProduct_uvu_weight_only(i_in1, i_in2, split_size, reshape_size, filter_ir_out=i_out, irrep_normalization=None, regroup_output=False)


    # # uvu
    tp = e3nn.o3.TensorProduct(i_in1,i_in2,i_out,inst_tuple,shared_weights=False, internal_weights=False) # path_normalization="none", normalization="none"
    tp_cuda = tp.to(device="cuda")

    # full tp -> linear
    # i_out = full_tp.irreps_out
    unique_cg = []
    nnz_cg_cnt = 0
    all_cg_cnt = 0
    cg_dummy = torch.zeros(i_in1.dim, i_in2.dim, uvuv_i_out.dim)
    cg_dummy_coverage = torch.zeros(i_in1.dim, i_in2.dim, uvuv_i_out.dim)
    for inst in uvuv_tp.instructions:
        i = inst.i_in1
        j = inst.i_in2
        k = inst.i_out

        mul_in1, ir_in1 = i_in1[i]
        mul_in2, ir_in2 = i_in2[j]
        mul_out, ir_out = uvuv_i_out[k]

        cg = e3nn.o3.wigner_3j(ir_in1.l, ir_in2.l, ir_out.l)
        unique_cg += list(cg.unique())
        all_cg_cnt+= cg.numel()
        nnz_cg_cnt += cg.count_nonzero()

        partial_mat_cg = torch.zeros(i_in1[i].dim, i_in2[j].dim, uvuv_i_out[k].dim)
        
        ## uvuv
        for u,v in itertools.product(range(mul_in1), range(mul_in2)):
            partial_mat_cg [u*ir_in1.dim:(u+1)*ir_in1.dim,
            v*ir_in2.dim:(v+1)*ir_in2.dim,
            (u*mul_in2+v)*ir_out.dim:(u*mul_in2+v+1)*ir_out.dim] = cg 

        cg_dummy[i_in1.slices()[i], i_in2.slices()[j], uvuv_i_out.slices()[k]] = partial_mat_cg
        cg_dummy_coverage[i_in1.slices()[i], i_in2.slices()[j], uvuv_i_out.slices()[k]] = 1

    cg_dummy_kij = cg_dummy.permute(2,0,1)
    cg_dummy_kij_flat = cg_dummy_kij.reshape(uvuv_i_out.dim, -1)
    per_out_fiber = cg_dummy_kij_flat.count_nonzero(dim=1)
    unique_cg_lookup = list(set([x.item() for x in unique_cg]))
    uni_cg_cnt = len(unique_cg_lookup)

    # get perout_w_idx and per_blk_k_dim
    w_idx = 0
    cnt = 0
    blk_size = 32
    perout_w_idx = []
    per_blk_k_dim = []

    for mul,ir in i_out:
        for _ in range(mul):
            for i in range(ir.dim):
                cnt+=1
                perout_w_idx.append(w_idx)
                if(cnt%32==0):
                    per_blk_k_dim.append(ir.dim)
            w_idx+=1

    # per block metadata
    per_blk_w_start_idx = []
    per_blk_w_end_idx = []
    per_blk_w_precnt = []

    blk_u_in1_idx = []
    blk_u_in2_idx = []

    per_blk_u_in1_idx_range = []
    per_blk_u_in2_idx_range = []
    per_blk_fiber_start_range = []

    per_fiber_local_idx = []
    per_fiber_global_idx = []

    max_u_in1_dim = 0
    max_u_in2_dim = 0 
    max_fiber_cnt = 0

    pattern_len_array = []
    stride_mul_array = []
    rem_cum_idx_array = []
    rem_cumval_array = []
    fiber_cnt_array = []

    step = 32
    i = 0
    prev_w_idx_cnt = 0
    prev_idx = -1

    while(i< uvuv_i_out.dim):
        local_k_idx = cg_dummy_kij[i:i+step].nonzero()[:,0]
        blk_in1_idx = cg_dummy_kij[i:i+step].nonzero()[:,1]
        blk_in2_idx = cg_dummy_kij[i:i+step].nonzero()[:,2]
        
        used_weight_idx = perout_w_idx[i:i+step]

        per_blk_w_start_idx.append(used_weight_idx[0])
        per_blk_w_end_idx.append(used_weight_idx[-1]+1)
        
        if (prev_idx == used_weight_idx[0]):
            per_blk_w_precnt.append(prev_w_idx_cnt)
        else:
            per_blk_w_precnt.append(0)
        prev_idx = used_weight_idx[-1]

        prev_w_idx_cnt=0
        for w_idx_val in used_weight_idx:
            if w_idx_val == used_weight_idx[-1]: prev_w_idx_cnt+=1


        u_in1_list = blk_in1_idx.unique().tolist()
        u_in2_list = blk_in2_idx.unique().tolist()

        max_u_in1_dim = max(max_u_in1_dim, len(u_in1_list))
        max_u_in2_dim = max(max_u_in2_dim, len(u_in2_list))
        max_fiber_cnt = max(max_fiber_cnt, len(blk_in1_idx))

        blk_u_in1_idx.append(u_in1_list)
        blk_u_in2_idx.append(u_in2_list)

        per_blk_u_in1_idx_range.append(len(u_in1_list))
        per_blk_u_in2_idx_range.append(len(u_in2_list))

        per_blk_fiber_start_range.append(len(blk_in1_idx))

        local_in1_idx = [u_in1_list.index(x.item()) for x in blk_in1_idx]
        local_in2_idx = [u_in2_list.index(x.item()) for x in blk_in2_idx]

        for l_a,l_b,l_k,g_a,g_b in zip(local_in1_idx,local_in2_idx, local_k_idx,blk_in1_idx,blk_in2_idx):
            real_k = i + l_k
            cg_idx = unique_cg_lookup.index(cg_dummy_kij[real_k,g_a,g_b])

            per_fiber_local_idx.append(l_a)
            per_fiber_local_idx.append(l_b)
            per_fiber_local_idx.append(cg_idx)
            per_fiber_local_idx.append(l_k)

            per_fiber_global_idx.append(g_a)
            per_fiber_global_idx.append(g_b)
            per_fiber_global_idx.append(cg_idx)
            per_fiber_global_idx.append(real_k)


        pattern = per_out_fiber[i:i+step].tolist()
        pattern_length = 0
        if ([pattern[0]]*len(pattern) == pattern):
            pattern_length=1
        elif (list(itertools.chain.from_iterable(([pattern[0:3]]*len(pattern))))[0:len(pattern)] == pattern):
            pattern_length=3
        elif (list(itertools.chain.from_iterable(([pattern[0:5]]*len(pattern))))[0:len(pattern)] == pattern):
            pattern_length=5
        elif (list(itertools.chain.from_iterable(([pattern[0:7]]*len(pattern))))[0:len(pattern)] == pattern):
            pattern_length=7
        elif (list(itertools.chain.from_iterable(([pattern[0:9]]*len(pattern))))[0:len(pattern)] == pattern):
            pattern_length=9
        elif (list(itertools.chain.from_iterable(([pattern[0:11]]*len(pattern))))[0:len(pattern)] == pattern):
            pattern_length=11
        else:
            print("bad")

        unit_pattern = pattern[0:pattern_length]

        pattern_len_array.append(pattern_length)
        stride_mul_array.append(sum(unit_pattern))
        rem_cum_idx_array.append(pattern_length)

        rem_cumval_array.append(cumsum_list(unit_pattern, False))
        fiber_cnt_array.append(unit_pattern)

        i += step

    rem_cum_idx_array = cumsum_list(rem_cum_idx_array,False)
    per_blk_u_in1_idx_range  = cumsum_list(per_blk_u_in1_idx_range)
    per_blk_u_in2_idx_range  = cumsum_list(per_blk_u_in2_idx_range)
    per_blk_fiber_start_range = cumsum_list(per_blk_fiber_start_range)

    rem_cumval_array = list(itertools.chain.from_iterable(rem_cumval_array))
    fiber_cnt_array = list(itertools.chain.from_iterable(fiber_cnt_array))
    blk_u_in1_idx = list(itertools.chain.from_iterable(blk_u_in1_idx))
    blk_u_in2_idx = list(itertools.chain.from_iterable(blk_u_in2_idx))

    per_fiber_local_idx = torch.tensor(per_fiber_local_idx).to(torch.uint8)
    per_fiber_global_idx = torch.tensor(per_fiber_global_idx)

    # per_blk_w_idx = []
    # for w_idx, i in enumerate(uvuv_i_out):
    #     for j in range(i.ir.dim):
    #         per_blk_w_idx.append(w_idx)


    uni_w3j = torch.tensor(unique_cg_lookup)

    if(tp.shared_weights):
        weight = torch.rand(tp.weight_numel)
    else:
        weight = torch.rand(batch_size,tp.weight_numel)
        # weight = torch.ones(batch_size,tp.weight_numel)
        # weight[:,928:960] = 2

    # rearrange weight
    weight_sptp = []
    for inst_idx in weight_mul.inv:
        u,v, slice = weight_mul.weight_uv_pair_sorted_slice[inst_idx]
        weight_sptp.append(weight[:,slice])
    weight_sptp = torch.cat(weight_sptp,dim=1)

    path_weight = torch.zeros(i_out.dim) 
    for inst in tp.instructions:
        k = inst.i_out
        path_weight[i_out.slices()[k]] = inst.path_weight

    
    out_cuda = torch.zeros((batch_size,uvuv_i_out.dim), device="cuda")

    
    weight_cuda = weight.to("cuda")
    weight_T_cuda = weight_sptp.to("cuda").T.contiguous()

    uni_w3j_cuda = uni_w3j.to("cuda")
    pw_cuda = path_weight.to("cuda")
    
    meta_cuda = to_cuda_list(
                        per_blk_w_start_idx,
                        per_blk_w_end_idx,
                        per_blk_w_precnt,
                        per_blk_k_dim,
                        per_blk_u_in1_idx_range,
                        per_blk_u_in2_idx_range,
                        per_blk_fiber_start_range,
                        
                        blk_u_in1_idx,
                        blk_u_in2_idx,

                        pattern_len_array,
                        stride_mul_array,
                        rem_cum_idx_array,
                        rem_cumval_array,
                        fiber_cnt_array, 
                        
                        per_fiber_local_idx,
                        input_dtype=torch.uint16)
    size_input = [max_u_in1_dim,max_u_in2_dim,max_fiber_cnt,uni_cg_cnt]
    size_input = [int(x) for x in size_input]

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(10):
        in1_cuda = torch.rand(batch_size, i_in1.dim, device="cuda")
        in2_cuda = torch.rand(batch_size, i_in2.dim, device="cuda")

        print(i)

        torch.cuda.nvtx.range_push(f"sptp")
        # full sptp
        in1_cuda_T = in1_cuda.T.contiguous()
        in2_cuda_T = in2_cuda.T.contiguous()
        torch.cuda.nvtx.range_push(f"sptp_linear")
        sptp.sptp_linear_v1(in1_cuda_T,in2_cuda_T, out_cuda, uni_w3j_cuda, weight_T_cuda,pw_cuda,*meta_cuda,*size_input)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        # original e3nn
        torch.cuda.nvtx.range_push(f"e3nn")
        e3nn_fc_out_rand = tp_cuda(in1_cuda,in2_cuda,weight_cuda)
        torch.cuda.nvtx.range_pop()
        if(use_compare):
            # print(out_cuda)
            # print(e3nn_fc_out_rand)
            compare(out_cuda[0].cpu(),e3nn_fc_out_rand[0].cpu())
    torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    main()
