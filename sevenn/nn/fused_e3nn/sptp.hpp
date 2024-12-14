#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
static void check(cudaError_t err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
static void checkLast(char const* const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void sptp_all_forward_linear_v1_cuda(
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
    size_t uni_cg_cnt);


void bwd_sptp_linear_cuda_v1(
    torch::Tensor in1, 
    torch::Tensor in2,
    torch::Tensor weight,
    torch::Tensor mem_dL_dO,
    torch::Tensor mem_dL_din1,
    torch::Tensor mem_dL_din2,
    torch::Tensor mem_dL_dW,
    torch::Tensor mem_debug,

    torch::Tensor t_in1_idxing,
    torch::Tensor t_in1_ival,
    torch::Tensor t_in1_related_path_idx,

    torch::Tensor t_path_array1,
    torch::Tensor t_path_array2,
    torch::Tensor t_per_path_fiber_start,
    torch::Tensor t_path_weight,
    torch::Tensor t_per_path_weight_pos,

    torch::Tensor t_fiber_array,
    torch::Tensor t_unique_cg_val,

    size_t path_cnt,
    size_t per_block_batch
    );


void bwd_sptp_linear_cuda_v1_shared(
    torch::Tensor in1, 
    torch::Tensor in2,
    torch::Tensor weight,
    torch::Tensor mem_dL_dO,
    torch::Tensor mem_dL_din1,
    torch::Tensor mem_dL_din2,
    torch::Tensor mem_dL_dW,
    torch::Tensor mem_debug,

    torch::Tensor t_in1_idxing,
    torch::Tensor t_in1_ival,
    torch::Tensor t_in1_related_path_idx,

    torch::Tensor t_path_array1,
    torch::Tensor t_path_array2,
    torch::Tensor t_per_path_fiber_start,
    torch::Tensor t_path_weight,
    torch::Tensor t_per_path_weight_pos,

    torch::Tensor t_fiber_array,
    torch::Tensor t_unique_cg_val,

    size_t path_cnt,
    size_t per_block_batch,
    size_t max_ir_dim
    );

void fwd_sptp_linear_cuda_v2(
    torch::Tensor in1, 
    torch::Tensor in2,
    torch::Tensor weight,
    torch::Tensor out,

    torch::Tensor t_in1_idxing,
    torch::Tensor t_in1_ival,
    torch::Tensor t_in1_related_path_idx,

    torch::Tensor t_path_array1,
    torch::Tensor t_path_array2,
    torch::Tensor t_per_path_fiber_start,
    torch::Tensor t_path_weight,
    torch::Tensor t_per_path_weight_pos,

    torch::Tensor t_fiber_array,
    torch::Tensor t_unique_cg_val,

    size_t path_cnt,
    size_t per_block_batch
    );

void fwd_sptp_linear_cuda_v2_shared(
    torch::Tensor in1, 
    torch::Tensor in2,
    torch::Tensor weight,
    torch::Tensor out,

    torch::Tensor t_in1_idxing,
    torch::Tensor t_in1_ival,
    torch::Tensor t_in1_related_path_idx,

    torch::Tensor t_path_array1,
    torch::Tensor t_path_array2,
    torch::Tensor t_per_path_fiber_start,
    torch::Tensor t_path_weight,
    torch::Tensor t_per_path_weight_pos,

    torch::Tensor t_fiber_array,
    torch::Tensor t_unique_cg_val,

    size_t path_cnt,
    size_t per_block_batch,
    size_t max_ir_dim
    );

void bwd_bwd_sptp_linear_cuda_v2_shared(
    torch::Tensor mem_dF_din1, 
    torch::Tensor mem_dF_din2,
    torch::Tensor mem_dF_dW,
    torch::Tensor mem_dE_dO,

    torch::Tensor in1,
    torch::Tensor in2,
    torch::Tensor weight,
    
    torch::Tensor mem_dF_dO,
    torch::Tensor mem_dL_dW,
    torch::Tensor mem_dL_din1,
    torch::Tensor mem_dL_din2,
    torch::Tensor mem_debug,

    torch::Tensor t_in1_idxing,
    torch::Tensor t_in1_ival,
    torch::Tensor t_in1_related_path_idx,

    torch::Tensor t_path_array1,
    torch::Tensor t_path_array2,
    torch::Tensor t_per_path_fiber_start,
    torch::Tensor t_path_weight,
    torch::Tensor t_per_path_weight_pos,

    torch::Tensor t_fiber_array,
    torch::Tensor t_unique_cg_val,

    size_t path_cnt,
    size_t per_block_batch,
    size_t max_ir_dim
);
