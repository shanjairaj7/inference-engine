// Fused CPU kernels + a single-call "megakernel" forward for Qwen2-style models.
// The whole decoder stack runs in one native call per engine step: no Python or
// per-op allocation between layers. GEMMs go to MKL/oneDNN via ATen; everything
// else (norms, RoPE + KV-cache write, paged attention, SiLU*mul) is fused here.
#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>
#include <fbgemm/Fbgemm.h>

using at::Tensor;

static inline float dot(const float* __restrict a, const float* __restrict b, int n) {
  float s = 0.f;
#pragma omp simd reduction(+ : s)
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

// out = rmsnorm(x) * w, row-wise. x/out: [T,H]
static void rmsnorm(const float* x, const float* w, float* out, int64_t T, int64_t H, float eps) {
#pragma omp parallel for schedule(static)
  for (int64_t t = 0; t < T; ++t) {
    const float* xr = x + t * H;
    float* o = out + t * H;
    float ss = dot(xr, xr, H);
    float inv = 1.f / std::sqrt(ss / H + eps);
#pragma omp simd
    for (int64_t i = 0; i < H; ++i) o[i] = xr[i] * inv * w[i];
  }
}

// Applies neox-style RoPE to q and k inside the fused qkv buffer and writes k,v
// into the paged cache at slot_mapping[t].
static void rope_store_kv(float* qkv, const int64_t* pos, const float* cos_sin, const int64_t* slots,
                          float* kc, float* vc, int64_t T, int nh, int nkv, int D) {
  const int half = D / 2;
  const int64_t row = (int64_t)(nh + 2 * nkv) * D;
#pragma omp parallel for schedule(static)
  for (int64_t t = 0; t < T; ++t) {
    float* r = qkv + t * row;
    const float* cs = cos_sin + pos[t] * D;  // [cos(half) | sin(half)]
    for (int h = 0; h < nh + nkv; ++h) {
      float* x = r + h * D;
#pragma omp simd
      for (int i = 0; i < half; ++i) {
        float x1 = x[i], x2 = x[i + half], c = cs[i], s = cs[half + i];
        x[i] = x1 * c - x2 * s;
        x[i + half] = x2 * c + x1 * s;
      }
    }
    int64_t slot = slots[t];
    std::copy(r + nh * D, r + (nh + nkv) * D, kc + slot * nkv * D);
    std::copy(r + (nh + nkv) * D, r + (nh + 2 * nkv) * D, vc + slot * nkv * D);
  }
}

// Paged attention over a flattened batch that mixes prefill chunks and decode tokens.
// seq_info[s] = {q_start, q_len, ctx_len}; query i of seq s sits at position ctx_len-q_len+i
// and attends causally to cache positions [0, pos]. GQA: the q heads sharing a kv head are
// processed together so every K/V row is loaded once per group.
static void paged_attention(const float* qkv, const float* kc, const float* vc, const int32_t* seq_info,
                            const int32_t* btab, int64_t max_blocks, int64_t S, int bs, int nh, int nkv,
                            int D, float* out) {
  const int G = nh / nkv;
  const int64_t row = (int64_t)(nh + 2 * nkv) * D;
  const float scale = 1.f / std::sqrt((float)D);
  std::vector<int64_t> work;  // (token index, kv head, seq)
  for (int64_t s = 0; s < S; ++s)
    for (int i = 0; i < seq_info[s * 3 + 1]; ++i)
      for (int kh = 0; kh < nkv; ++kh) work.push_back(((int64_t)s << 40) | ((int64_t)i << 8) | kh);
#pragma omp parallel
  {
    std::vector<float> sc, acc(G * D);
#pragma omp for schedule(dynamic, 4)
    for (size_t w = 0; w < work.size(); ++w) {
      int64_t s = work[w] >> 40;
      int i = (work[w] >> 8) & 0xffffffff;
      int kh = work[w] & 0xff;
      int q_start = seq_info[s * 3], q_len = seq_info[s * 3 + 1], ctx = seq_info[s * 3 + 2];
      int64_t t = q_start + i;
      int npos = ctx - q_len + i + 1;
      const int32_t* bt = btab + s * max_blocks;
      const float* q = qkv + t * row + kh * G * D;
      sc.resize((size_t)G * npos);
      float mx[64];
      for (int g = 0; g < G; ++g) mx[g] = -INFINITY;
      for (int j = 0; j < npos; ++j) {
        const float* k = kc + (((int64_t)bt[j / bs] * bs + j % bs) * nkv + kh) * D;
        for (int g = 0; g < G; ++g) {
          float v = dot(q + g * D, k, D) * scale;
          sc[g * npos + j] = v;
          mx[g] = std::max(mx[g], v);
        }
      }
      float inv[64];
      for (int g = 0; g < G; ++g) {
        float sum = 0.f;
        float* p = sc.data() + g * npos;
        for (int j = 0; j < npos; ++j) { p[j] = std::exp(p[j] - mx[g]); sum += p[j]; }
        inv[g] = 1.f / sum;
      }
      std::fill(acc.begin(), acc.end(), 0.f);
      for (int j = 0; j < npos; ++j) {
        const float* v = vc + (((int64_t)bt[j / bs] * bs + j % bs) * nkv + kh) * D;
        for (int g = 0; g < G; ++g) {
          float p = sc[g * npos + j];
          float* a = acc.data() + g * D;
#pragma omp simd
          for (int d = 0; d < D; ++d) a[d] += p * v[d];
        }
      }
      float* o = out + t * nh * D + kh * G * D;
      for (int g = 0; g < G; ++g)
        for (int d = 0; d < D; ++d) o[g * D + d] = acc[g * D + d] * inv[g];
    }
  }
}

// gu: [T, 2I] = [gate | up] -> out[T, I] = silu(gate) * up
static void silu_mul(const float* gu, float* out, int64_t T, int64_t I) {
#pragma omp parallel for schedule(static)
  for (int64_t t = 0; t < T; ++t) {
    const float* g = gu + t * 2 * I;
    const float* u = g + I;
    float* o = out + t * I;
#pragma omp simd
    for (int64_t i = 0; i < I; ++i) o[i] = g[i] / (1.f + std::exp(-g[i])) * u[i];
  }
}


// ---------------------------------------------------------------------------
// W8A8 int8 linear: weights symmetric per-output-channel int8, activations asymmetric
// per-token 7-bit (u7 x s8 pairs can't saturate AVX2's 16-bit vpmaddubsw accumulation).
// GEMM runs on fbgemm's packed AVX2 kernels with raw int32 output; dequantization is
// fused with whatever comes next (bias, residual add, or SiLU(gate)*up).
// y[i,j] = sa[i] * sw[j] * (C[i,j] - za[i] * colsum(Wq[j])) + b[j]
struct QLinear {
  int N, K;
  std::unique_ptr<fbgemm::PackBMatrix<int8_t>> B;
  std::vector<float> sw, bias;
  std::vector<int32_t> colsum;
};
static std::vector<std::unique_ptr<QLinear>> g_qlin;
enum Epilogue { STORE = 0, ACCUM = 1, SILU_MUL = 2 };

int64_t pack_int8(Tensor w, c10::optional<Tensor> bias) {
  w = w.contiguous().to(at::kFloat);
  auto q = std::make_unique<QLinear>();
  const int N = w.size(0), K = w.size(1);
  q->N = N; q->K = K;
  q->sw.resize(N); q->colsum.assign(N, 0);
  std::vector<int8_t> wq((size_t)N * K);
  const float* W = w.data_ptr<float>();
  for (int n = 0; n < N; ++n) {
    float amax = 0.f;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(W[(int64_t)n * K + k]));
    float s = amax > 0 ? amax / 127.f : 1.f;
    q->sw[n] = s;
    for (int k = 0; k < K; ++k) {
      int v = (int)std::nearbyint(W[(int64_t)n * K + k] / s);
      v = std::min(127, std::max(-127, v));
      wq[(int64_t)n * K + k] = (int8_t)v;
      q->colsum[n] += v;
    }
  }
  q->B = std::make_unique<fbgemm::PackBMatrix<int8_t>>(fbgemm::matrix_op_t::Transpose, K, N, wq.data(), K, nullptr, 1);
  if (bias.has_value()) {
    Tensor b = bias->contiguous().to(at::kFloat);
    q->bias.assign(b.data_ptr<float>(), b.data_ptr<float>() + N);
  } else {
    q->bias.assign(N, 0.f);
  }
  g_qlin.push_back(std::move(q));
  return (int64_t)g_qlin.size() - 1;
}

static std::vector<uint8_t> g_aq;
static std::vector<int32_t> g_acc, g_za;
static std::vector<float> g_sa;

static void qlinear(const QLinear& L, const float* x, int64_t M, float* out, int epi) {
  const int K = L.K, N = L.N;
  if ((int64_t)g_aq.size() < M * K) g_aq.resize(M * K);
  if ((int64_t)g_acc.size() < M * N) g_acc.resize(M * N);
  if ((int64_t)g_sa.size() < M) { g_sa.resize(M); g_za.resize(M); }
  uint8_t* aq = g_aq.data();
  int32_t* C = g_acc.data();
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < M; ++i) {
    const float* r = x + i * K;
    float mn = 0.f, mx = 0.f;
#pragma omp simd reduction(min : mn) reduction(max : mx)
    for (int k = 0; k < K; ++k) { mn = std::min(mn, r[k]); mx = std::max(mx, r[k]); }
    float s = (mx - mn) / 127.f;
    if (s == 0.f) s = 1.f;
    const float inv = 1.f / s;
    const int zp = (int)std::nearbyint(-mn * inv);
    g_sa[i] = s; g_za[i] = zp;
    uint8_t* q = aq + i * K;
#pragma omp simd
    for (int k = 0; k < K; ++k) {
      float v = std::nearbyint(r[k] * inv) + zp;
      q[k] = (uint8_t)std::min(127.f, std::max(0.f, v));
    }
  }
#pragma omp parallel
  {
    fbgemm::PackAMatrix<uint8_t> A(fbgemm::matrix_op_t::NoTranspose, (int)M, K, aq, K, nullptr, 1);
    fbgemm::DoNothing<int32_t, int32_t> nop;
    fbgemm::memCopy<> out_proc(nop);
    fbgemm::fbgemmPacked(A, *L.B, C, C, (uint32_t)N, out_proc, omp_get_thread_num(), omp_get_num_threads());
  }
  const float* sw = L.sw.data(); const float* b = L.bias.data(); const int32_t* cs = L.colsum.data();
  if (epi == SILU_MUL) {
    const int I = N / 2;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; ++i) {
      const float sa = g_sa[i]; const int za = g_za[i];
      const int32_t* c = C + i * N;
      float* o = out + i * I;
#pragma omp simd
      for (int j = 0; j < I; ++j) {
        float g = sa * sw[j] * (float)(c[j] - za * cs[j]) + b[j];
        float u = sa * sw[j + I] * (float)(c[j + I] - za * cs[j + I]) + b[j + I];
        o[j] = g / (1.f + std::exp(-g)) * u;
      }
    }
    return;
  }
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < M; ++i) {
    const float sa = g_sa[i]; const int za = g_za[i];
    const int32_t* c = C + i * N;
    float* o = out + i * N;
    if (epi == ACCUM) {
#pragma omp simd
      for (int j = 0; j < N; ++j) o[j] += sa * sw[j] * (float)(c[j] - za * cs[j]) + b[j];
    } else {
#pragma omp simd
      for (int j = 0; j < N; ++j) o[j] = sa * sw[j] * (float)(c[j] - za * cs[j]) + b[j];
    }
  }
}

Tensor forward(Tensor input_ids, Tensor positions, Tensor slots, Tensor seq_info, Tensor block_tables,
               Tensor logits_idx, Tensor embed, std::vector<Tensor> weights, Tensor final_norm,
               std::vector<Tensor> k_caches, std::vector<Tensor> v_caches, Tensor cos_sin,
               int64_t nh, int64_t nkv, int64_t D, int64_t bs, double eps, std::vector<int64_t> q) {
  const int64_t T = input_ids.size(0), H = embed.size(1), S = seq_info.size(0);
  const int64_t nl = k_caches.size();
  const bool quant = !q.empty();
  const int64_t I = (quant && q[2] >= 0) ? g_qlin[q[2]]->N / 2 : weights[5].size(0) / 2;
  Tensor qkv = at::empty({T, (nh + 2 * nkv) * D}, embed.options());
  Tensor x = embed.index_select(0, input_ids);  // residual stream [T,H]
  Tensor h = at::empty_like(x);
  Tensor attn = at::empty({T, nh * D}, x.options());
  Tensor act = at::empty({T, I}, x.options());
  const int64_t* pos = positions.data_ptr<int64_t>();
  const int64_t* sl = slots.data_ptr<int64_t>();
  const int32_t* si = seq_info.data_ptr<int32_t>();
  const int32_t* bt = block_tables.data_ptr<int32_t>();
  for (int64_t l = 0; l < nl; ++l) {
    const Tensor* W = &weights[l * 7];  // ln1, wqkv, bqkv, wo, ln2, wgu, wdown
    const int64_t* Q = quant ? &q[l * 4] : nullptr;  // qkv, o, gate_up, down
    rmsnorm(x.data_ptr<float>(), W[0].data_ptr<float>(), h.data_ptr<float>(), T, H, eps);
    if (quant && Q[0] >= 0) qlinear(*g_qlin[Q[0]], h.data_ptr<float>(), T, qkv.data_ptr<float>(), STORE);
    else at::addmm_out(qkv, W[2], h, W[1].t());
    rope_store_kv(qkv.data_ptr<float>(), pos, cos_sin.data_ptr<float>(), sl, k_caches[l].data_ptr<float>(),
                  v_caches[l].data_ptr<float>(), T, nh, nkv, D);
    paged_attention(qkv.data_ptr<float>(), k_caches[l].data_ptr<float>(), v_caches[l].data_ptr<float>(), si, bt,
                    block_tables.size(1), S, bs, nh, nkv, D, attn.data_ptr<float>());
    if (quant && Q[1] >= 0) qlinear(*g_qlin[Q[1]], attn.data_ptr<float>(), T, x.data_ptr<float>(), ACCUM);
    else x.addmm_(attn, W[3].t());
    rmsnorm(x.data_ptr<float>(), W[4].data_ptr<float>(), h.data_ptr<float>(), T, H, eps);
    if (quant && Q[2] >= 0) {
      qlinear(*g_qlin[Q[2]], h.data_ptr<float>(), T, act.data_ptr<float>(), SILU_MUL);
    } else {
      Tensor gu = at::mm(h, W[5].t());
      silu_mul(gu.data_ptr<float>(), act.data_ptr<float>(), T, I);
    }
    if (quant && Q[3] >= 0) qlinear(*g_qlin[Q[3]], act.data_ptr<float>(), T, x.data_ptr<float>(), ACCUM);
    else x.addmm_(act, W[6].t());
  }
  Tensor last = x.index_select(0, logits_idx).contiguous();
  Tensor ln = at::empty_like(last);
  rmsnorm(last.data_ptr<float>(), final_norm.data_ptr<float>(), ln.data_ptr<float>(), last.size(0), H, eps);
  if (quant && q[nl * 4] >= 0) {
    const QLinear& lm = *g_qlin[q[nl * 4]];
    Tensor logits = at::empty({last.size(0), (int64_t)lm.N}, x.options());
    qlinear(lm, ln.data_ptr<float>(), last.size(0), logits.data_ptr<float>(), STORE);
    return logits;
  }
  return at::mm(ln, embed.t());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, "fused Qwen2 forward over paged KV");
  m.def("pack_int8", &pack_int8, "quantize + fbgemm-pack a [N,K] linear weight; returns a handle");
}
