#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template<typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
  scalar_t* __restrict__ arr,
  const float* __restrict__ cos_ptr,
  const float* __restrict__ sin_ptr,
  int rot_offset,
  int embed_dim)
{
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template<typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
  scalar_t* __restrict__ query,                 // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key,                   // [num_tokens, num_heads, head_size]
  const float* __restrict__ cos_cache,   // [max_position, 1, rot_dim]
  const float* __restrict__ sin_cache,   // [max_position, 1, rot_dim]
  const int rot_dim,
  const int query_stride,
  const int key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;

  const int embed_dim = rot_dim;
  const float* cos_ptr = cos_cache;
  const float* sin_ptr = sin_cache;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }
}

} // namespace vllm

void rotary_embedding(
  torch::Tensor& query,             // [num_tokens, num_heads, head_size]
  torch::Tensor& key,               // [num_tokens, num_kv_heads, head_size]
  int head_size,
  torch::Tensor& cos_cache,     // [max_position, 1, rot_dim]
  torch::Tensor& sin_cache,     // [max_position, 1, rot_dim]
  bool is_neox) {
  int num_tokens = query.size(0);
  int rot_dim = cos_cache.size(2);
  int num_heads = query.size(1);
  int num_kv_heads = key.size(1);
  int query_stride = query.stride(0);
  int key_stride = key.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Here we cast cos_cache and sin_cache to float, following what is done in flash-attn implementation of ROPE.
  VLLM_DISPATCH_FLOATING_TYPES(
    query.scalar_type(),
    "rotary_embedding",
    [&] {
      if (is_neox) {
        vllm::rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_cache.data_ptr<float>(),
          sin_cache.data_ptr<float>(),
          rot_dim,
          query_stride,
          key_stride,
          num_heads,
          num_kv_heads,
          head_size);
      } else {
        vllm::rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_cache.data_ptr<float>(),
          sin_cache.data_ptr<float>(),
          rot_dim,
          query_stride,
          key_stride,
          num_heads,
          num_kv_heads,
          head_size);
      }
    });
}
