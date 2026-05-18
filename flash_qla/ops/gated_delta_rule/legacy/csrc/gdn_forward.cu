#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(status));
  }
}

__device__ __forceinline__ float subgroup_sum_lane0(float value,
                                                    int width) {
  constexpr unsigned mask = 0xffffffffU;
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, width);
  }
  return value;
}

__device__ __forceinline__ float subgroup_broadcast_lane0(float value,
                                                          int width) {
  return __shfl_sync(0xffffffffU, value, 0, width);
}

template <int D, int COLS, int WIDTH>
__global__ void gdn_forward_kernel(const float* __restrict__ q,
                                   const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   const float* __restrict__ gate,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ initial_state,
                                   float* __restrict__ output,
                                   float* __restrict__ final_state,
                                   int batch,
                                   int tokens,
                                   int q_heads,
                                   int v_heads,
                                   float scale) {
  static_assert(D % (COLS * (32 / WIDTH)) == 0);
  constexpr int subgroups_per_warp = 32 / WIDTH;
  constexpr int rows_per_lane = (D + WIDTH - 1) / WIDTH;

  const int hv = blockIdx.x;
  const int b = blockIdx.y;
  const int subgroup = threadIdx.x / WIDTH;
  const int lane = threadIdx.x % WIDTH;
  const int group_base =
      (blockIdx.z * blockDim.y + threadIdx.y) * subgroups_per_warp + subgroup;
  const int col_base = group_base * COLS;
  const int hq = hv / (v_heads / q_heads);

  float state_shard[COLS][rows_per_lane];

#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      float value = 0.0F;
      if (row < D) {
        const auto state_index =
            (((static_cast<int64_t>(b) * v_heads + hv) * D + col) * D) + row;
        value = initial_state == nullptr ? 0.0F : initial_state[state_index];
      }
      state_shard[c][r] = value;
    }
  }

  for (int t = 0; t < tokens; ++t) {
    const auto gate_index =
        ((static_cast<int64_t>(b) * tokens + t) * v_heads + hv);
    float gate_value = 0.0F;
    float beta_value = 0.0F;
    if (threadIdx.x == 0) {
      gate_value = __expf(gate[gate_index]);
      beta_value = beta[gate_index];
    }
    gate_value = __shfl_sync(0xffffffffU, gate_value, 0);
    beta_value = __shfl_sync(0xffffffffU, beta_value, 0);

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float kv_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      kv_partial[c] = 0.0F;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      float q_value = 0.0F;
      float k_value = 0.0F;
      if (row < D) {
        const auto qk_index =
            (((static_cast<int64_t>(b) * tokens + t) * q_heads + hq) * D) + row;
        q_value = q[qk_index];
        k_value = k[qk_index];
      }
      q_reg[r] = q_value;
      k_reg[r] = k_value;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        kv_partial[c] += state_shard[c][r] * k_value;
      }
    }

    float delta[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const float kv_col = subgroup_sum_lane0(kv_partial[c], WIDTH);
      float delta_value = 0.0F;
      if (lane == 0) {
        const auto v_index =
            (((static_cast<int64_t>(b) * tokens + t) * v_heads + hv) * D) +
            col_base + c;
        delta_value = (v[v_index] - gate_value * kv_col) * beta_value;
      }
      delta[c] = subgroup_broadcast_lane0(delta_value, WIDTH);
    }

    float attn_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = 0.0F;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const float new_state =
            fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
        state_shard[c][r] = new_state;
        attn_partial[c] += new_state * q_reg[r];
      }
    }

#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = subgroup_sum_lane0(attn_partial[c], WIDTH);
    }

    if (lane == 0) {
      const auto out_base =
          (((static_cast<int64_t>(b) * tokens + t) * v_heads + hv) * D);
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        output[out_base + col_base + c] = attn_partial[c] * scale;
      }
    }
  }

#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      if (row < D) {
        const auto state_index =
            (((static_cast<int64_t>(b) * v_heads + hv) * D + col) * D) + row;
        final_state[state_index] = state_shard[c][r];
      }
    }
  }
}

template <int D>
void launch_gdn_forward(const float* q,
                        const float* k,
                        const float* v,
                        const float* gate,
                        const float* beta,
                        const float* initial_state,
                        float* output,
                        float* final_state,
                        int batch,
                        int tokens,
                        int q_heads,
                        int v_heads,
                        float scale,
                        cudaStream_t stream) {
  constexpr int cols = D == 128 ? 4 : 1;
  constexpr int width = D == 128 ? 16 : 32;
  constexpr int groups_per_warp = 32 / width;
  constexpr int column_groups_per_block = 8;
  const dim3 block(32, column_groups_per_block);
  const int groups = D / cols;
  const int z = (groups + column_groups_per_block * groups_per_warp - 1) /
                (column_groups_per_block * groups_per_warp);
  const dim3 grid(v_heads, batch, z);
  gdn_forward_kernel<D, cols, width>
      <<<grid, block, 0, stream>>>(q,
                                   k,
                                   v,
                                   gate,
                                   beta,
                                   initial_state,
                                   output,
                                   final_state,
                                   batch,
                                   tokens,
                                   q_heads,
                                   v_heads,
                                   scale);
}

void validate_tensor(const torch::Tensor& tensor,
                     const char* name,
                     int64_t dims) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name,
              " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == dims, name, " has wrong rank");
}

}  // namespace

std::vector<torch::Tensor> gdn_forward(torch::Tensor q,
                                       torch::Tensor k,
                                       torch::Tensor v,
                                       torch::Tensor gate,
                                       torch::Tensor beta,
                                       c10::optional<torch::Tensor> initial_state,
                                       double scale) {
  validate_tensor(q, "q", 4);
  validate_tensor(k, "k", 4);
  validate_tensor(v, "v", 4);
  validate_tensor(gate, "gate", 3);
  validate_tensor(beta, "beta", 3);

  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
  const int batch = static_cast<int>(q.size(0));
  const int tokens = static_cast<int>(q.size(1));
  const int q_heads = static_cast<int>(q.size(2));
  const int dim = static_cast<int>(q.size(3));
  const int v_heads = static_cast<int>(v.size(2));
  TORCH_CHECK(v.size(0) == batch && v.size(1) == tokens && v.size(3) == dim,
              "v must have shape [B, T, Hv, D] matching q/k");
  TORCH_CHECK(gate.size(0) == batch && gate.size(1) == tokens &&
                  gate.size(2) == v_heads,
              "gate must have shape [B, T, Hv]");
  TORCH_CHECK(beta.sizes() == gate.sizes(),
              "beta must have the same shape as gate");
  TORCH_CHECK(v_heads % q_heads == 0, "Hv must be divisible by Hq");
  TORCH_CHECK(dim == 16 || dim == 32 || dim == 64 || dim == 128,
              "D must be one of 16, 32, 64, or 128");

  const float* initial_ptr = nullptr;
  if (initial_state.has_value() && initial_state.value().defined()) {
    const auto& h0 = initial_state.value();
    validate_tensor(h0, "initial_state", 4);
    TORCH_CHECK(h0.size(0) == batch && h0.size(1) == v_heads &&
                    h0.size(2) == dim && h0.size(3) == dim,
                "initial_state must have shape [B, Hv, D, D]");
    initial_ptr = h0.data_ptr<float>();
  }

  auto output = torch::empty_like(v);
  auto final_state = torch::empty({batch, v_heads, dim, dim}, q.options());

  const auto stream = at::cuda::getCurrentCUDAStream(q.device().index()).stream();
  switch (dim) {
    case 16:
      launch_gdn_forward<16>(q.data_ptr<float>(),
                             k.data_ptr<float>(),
                             v.data_ptr<float>(),
                             gate.data_ptr<float>(),
                             beta.data_ptr<float>(),
                             initial_ptr,
                             output.data_ptr<float>(),
                             final_state.data_ptr<float>(),
                             batch,
                             tokens,
                             q_heads,
                             v_heads,
                             static_cast<float>(scale),
                             stream);
      break;
    case 32:
      launch_gdn_forward<32>(q.data_ptr<float>(),
                             k.data_ptr<float>(),
                             v.data_ptr<float>(),
                             gate.data_ptr<float>(),
                             beta.data_ptr<float>(),
                             initial_ptr,
                             output.data_ptr<float>(),
                             final_state.data_ptr<float>(),
                             batch,
                             tokens,
                             q_heads,
                             v_heads,
                             static_cast<float>(scale),
                             stream);
      break;
    case 64:
      launch_gdn_forward<64>(q.data_ptr<float>(),
                             k.data_ptr<float>(),
                             v.data_ptr<float>(),
                             gate.data_ptr<float>(),
                             beta.data_ptr<float>(),
                             initial_ptr,
                             output.data_ptr<float>(),
                             final_state.data_ptr<float>(),
                             batch,
                             tokens,
                             q_heads,
                             v_heads,
                             static_cast<float>(scale),
                             stream);
      break;
    case 128:
      launch_gdn_forward<128>(q.data_ptr<float>(),
                              k.data_ptr<float>(),
                              v.data_ptr<float>(),
                              gate.data_ptr<float>(),
                              beta.data_ptr<float>(),
                              initial_ptr,
                              output.data_ptr<float>(),
                              final_state.data_ptr<float>(),
                              batch,
                              tokens,
                              q_heads,
                              v_heads,
                              static_cast<float>(scale),
                              stream);
      break;
  }
  check_cuda(cudaGetLastError(), "gdn_forward launch");
  return {output, final_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_forward", &gdn_forward, "SM70/SM75 legacy GDN forward");
}
