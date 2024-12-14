#include <torch/extension.h>
#include "sptp.hpp"
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



void sptp_linear_v1(    
    torch::Tensor in1, 
    torch::Tensor in2,
    torch::Tensor out, 
    torch::Tensor uni_w3j, 
    torch::Tensor weight, 
    torch::Tensor path_weight,

    torch::Tensor per_blk_w_start_idx,
    torch::Tensor per_blk_w_end_idx,
    torch::Tensor per_blk_w_precnt,
    torch::Tensor per_blk_k_dim,

    torch::Tensor per_blk_u_in1_idx_range,
    torch::Tensor per_blk_u_in2_idx_range,
    torch::Tensor per_blk_fiber_start_range,
    torch::Tensor blk_u_in1_idx,
    torch::Tensor blk_u_in2_idx,

    torch::Tensor pattern_len_array,
    torch::Tensor stride_mul_array,
    torch::Tensor rem_cum_idx_array,
    torch::Tensor rem_cumval_array,
    torch::Tensor fiber_cnt_array,

    torch::Tensor per_fiber_local_idx,

    size_t max_unique_in1_size, 
    size_t max_unique_in2_size, 
    size_t max_fiber_cnt,
    size_t uni_cg_cnt){

    CHECK_INPUT(in1);
    CHECK_INPUT(in2);
    CHECK_INPUT(out);
    CHECK_INPUT(uni_w3j);
    CHECK_INPUT(weight);
    CHECK_INPUT(path_weight);

    CHECK_INPUT(per_blk_w_start_idx);
    CHECK_INPUT(per_blk_w_end_idx);
    CHECK_INPUT(per_blk_w_precnt);
    CHECK_INPUT(per_blk_k_dim);

    CHECK_INPUT(per_blk_u_in1_idx_range);
    CHECK_INPUT(per_blk_u_in2_idx_range);
    CHECK_INPUT(per_blk_fiber_start_range);
    CHECK_INPUT(blk_u_in1_idx);
    CHECK_INPUT(blk_u_in2_idx);

    CHECK_INPUT(pattern_len_array);
    CHECK_INPUT(stride_mul_array);
    CHECK_INPUT(rem_cum_idx_array);
    CHECK_INPUT(rem_cumval_array);
    CHECK_INPUT(fiber_cnt_array);
    
    CHECK_INPUT(per_fiber_local_idx);

    sptp_all_forward_linear_v1_cuda(
        in1,  in2,
        out,  uni_w3j,  weight,  path_weight,
        
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

        max_unique_in1_size, 
        max_unique_in2_size, 
        max_fiber_cnt,
        uni_cg_cnt);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sptp_linear_v1", &sptp_linear_v1, "sptp_linear_v1");
}
