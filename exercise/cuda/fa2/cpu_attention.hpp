#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// Non-causal attention, Q/K/V/O layout: [B, H, L, HD], LSU: [B, H, L].
// Compute one full softmax row at a time, independently of GPU tile sizes.
inline void attention_cpu(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, std::vector<float>& o,
                          std::vector<float>& lsu, int B, int H, int L, int HD) {
    o.resize(q.size());
    lsu.resize(static_cast<std::size_t>(B) * H * L);
    std::vector<double> scores(L), output(HD);
    const double scale = 1.0 / std::sqrt(static_cast<double>(HD));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const std::size_t head = (static_cast<std::size_t>(b) * H + h) * L;
            const std::size_t base = head * HD;
            for (int row = 0; row < L; ++row) {
                const std::size_t q_row = base + static_cast<std::size_t>(row) * HD;
                for (int col = 0; col < L; ++col) {
                    const std::size_t k_row = base + static_cast<std::size_t>(col) * HD;
                    double dot = 0.0;
                    for (int d = 0; d < HD; ++d) {
                        dot += static_cast<double>(q[q_row + d]) * k[k_row + d];
                    }
                    scores[col] = dot * scale;
                }
                const double row_max = *std::max_element(scores.begin(), scores.end());
                double row_sum = 0.0;
                std::fill(output.begin(), output.end(), 0.0);
                for (int col = 0; col < L; ++col) {
                    const double weight = std::exp(scores[col] - row_max);
                    row_sum += weight;
                    const std::size_t v_row = base + static_cast<std::size_t>(col) * HD;
                    for (int d = 0; d < HD; ++d) {
                        output[d] += weight * v[v_row + d];
                    }
                }
                for (int d = 0; d < HD; ++d) {
                    o[q_row + d] = static_cast<float>(output[d] / row_sum);
                }
                lsu[head + row] = static_cast<float>(row_max + std::log(row_sum));
            }
        }
    }
}

inline bool check_attention(const char* name, const std::vector<float>& gpu,
                            const std::vector<float>& cpu, int H, int L, int width) {
    constexpr double atol = 1e-5, rtol = 1e-4;
    if (gpu.size() != cpu.size()) {
        std::fprintf(stderr, "FAIL %s: result sizes differ\n", name);
        return false;
    }
    double max_error = 0.0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(gpu[i]) - cpu[i]);
        const double tolerance = atol + rtol * std::fabs(static_cast<double>(cpu[i]));
        if (!std::isfinite(gpu[i]) || !std::isfinite(cpu[i]) || diff > tolerance) {
            const std::size_t row = i / width;
            std::fprintf(stderr,
                         "FAIL %s at [b=%zu, h=%zu, row=%zu, d=%zu]: "
                         "GPU=%.9g, CPU=%.9g, abs_error=%.9g (tolerance=%.9g)\n",
                         name, row / (H * L), (row / L) % H, row % L, i % width,
                         gpu[i], cpu[i], diff, tolerance);
            return false;
        }
        max_error = std::max(max_error, diff);
    }
    std::printf("PASS %s: all %zu elements match CPU results "
                "(max_abs_error=%.9g, atol=%.1e, rtol=%.1e)\n",
                name, gpu.size(), max_error, atol, rtol);
    return true;
}
