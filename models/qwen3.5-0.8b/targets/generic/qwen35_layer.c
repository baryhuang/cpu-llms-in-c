#include "qwen35_layer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Mirrors compiler/generate_qwen35_layer_fixture.py: double accumulation with
 * float casts at exactly the stored points, so declared boundaries match the
 * oracle-verified reference bit for bit. */

static void linear_forward(const float *inputs, const float *weights,
                           uint32_t rows, uint32_t input_size, uint32_t output_size,
                           float *outputs)
{
    for (uint32_t row = 0; row < rows; ++row) {
        const float *in = inputs + (size_t)row * input_size;
        for (uint32_t out = 0; out < output_size; ++out) {
            const float *weight = weights + (size_t)out * input_size;
            double total = 0.0;
            for (uint32_t index = 0; index < input_size; ++index) {
                total += (double)in[index] * weight[index];
            }
            outputs[(size_t)row * output_size + out] = (float)total;
        }
    }
}

static void rms_norm_one_plus(const float *inputs, const float *weight,
                              uint32_t rows, uint32_t width, float epsilon,
                              float *outputs)
{
    for (uint32_t row = 0; row < rows; ++row) {
        const float *values = inputs + (size_t)row * width;
        double mean_squared = 0.0;
        for (uint32_t index = 0; index < width; ++index) {
            mean_squared += (double)values[index] * values[index];
        }
        mean_squared = mean_squared / width + epsilon;
        const double inverse_root = pow(mean_squared, -0.5);
        for (uint32_t index = 0; index < width; ++index) {
            outputs[(size_t)row * width + index] =
                (float)((double)values[index] * inverse_root * (1.0 + (double)weight[index]));
        }
    }
}

static double silu(double value)
{
    return value / (1.0 + exp(-value));
}

static double sigmoid(double value)
{
    return 1.0 / (1.0 + exp(-value));
}

static double softplus(double value)
{
    return log1p(exp(value));
}

int qwen35_deltanet_layer_forward(const struct qwen35_layer_config *config,
                                  const struct qwen35_deltanet_weights *weights,
                                  const float *input,
                                  const struct qwen35_deltanet_outputs *outputs)
{
    const uint32_t seq = config->sequence_length;
    const uint32_t hidden = config->hidden_size;
    const uint32_t k_heads = config->linear_k_heads;
    const uint32_t v_heads = config->linear_v_heads;
    const uint32_t k_dim = config->linear_k_dim;
    const uint32_t v_dim = config->linear_v_dim;
    const uint32_t kernel = config->conv_kernel;
    const uint32_t key_dim = k_heads * k_dim;
    const uint32_t value_dim = v_heads * v_dim;
    const uint32_t conv_dim = 2U * key_dim + value_dim;
    const uint32_t intermediate = config->intermediate_size;
    int result = -1;

    float *mixed = malloc((size_t)seq * conv_dim * sizeof(float));
    float *z = malloc((size_t)seq * value_dim * sizeof(float));
    float *b = malloc((size_t)seq * v_heads * sizeof(float));
    float *a = malloc((size_t)seq * v_heads * sizeof(float));
    float *queries = malloc((size_t)seq * key_dim * sizeof(float));
    float *keys = malloc((size_t)seq * key_dim * sizeof(float));
    float *values = malloc((size_t)seq * value_dim * sizeof(float));
    double *state = calloc((size_t)v_heads * k_dim * v_dim, sizeof(double));
    float *mlp_normed = malloc((size_t)seq * hidden * sizeof(float));
    float *gate_buffer = malloc((size_t)seq * intermediate * sizeof(float));
    float *up_buffer = malloc((size_t)seq * intermediate * sizeof(float));
    float *mlp_out = malloc((size_t)seq * hidden * sizeof(float));
    if (!mixed || !z || !b || !a || !queries || !keys || !values || !state ||
        !mlp_normed || !gate_buffer || !up_buffer || !mlp_out) {
        goto cleanup;
    }

    rms_norm_one_plus(input, weights->input_norm, seq, hidden, config->rms_epsilon,
                      outputs->normed);
    linear_forward(outputs->normed, weights->in_proj_qkv, seq, hidden, conv_dim, mixed);
    linear_forward(outputs->normed, weights->in_proj_z, seq, hidden, value_dim, z);
    linear_forward(outputs->normed, weights->in_proj_b, seq, hidden, v_heads, b);
    linear_forward(outputs->normed, weights->in_proj_a, seq, hidden, v_heads, a);

    for (uint32_t row = 0; row < seq; ++row) {
        for (uint32_t channel = 0; channel < conv_dim; ++channel) {
            double total = 0.0;
            for (uint32_t tap = 0; tap < kernel; ++tap) {
                const int64_t source = (int64_t)row - (int64_t)(kernel - 1U) + (int64_t)tap;
                if (source >= 0) {
                    total += (double)mixed[(size_t)source * conv_dim + channel] *
                             weights->conv[(size_t)channel * kernel + tap];
                }
            }
            outputs->post_conv[(size_t)row * conv_dim + channel] = (float)silu(total);
        }
    }

    const double query_scale = pow((double)k_dim, -0.5);
    for (uint32_t row = 0; row < seq; ++row) {
        const float *post = outputs->post_conv + (size_t)row * conv_dim;
        for (uint32_t head = 0; head < k_heads; ++head) {
            const float *q_in = post + (size_t)head * k_dim;
            const float *k_in = post + key_dim + (size_t)head * k_dim;
            double q_squares = 0.0;
            double k_squares = 0.0;
            for (uint32_t index = 0; index < k_dim; ++index) {
                q_squares += (double)q_in[index] * q_in[index];
                k_squares += (double)k_in[index] * k_in[index];
            }
            const double q_inverse = pow(q_squares + 1.0e-6, -0.5);
            const double k_inverse = pow(k_squares + 1.0e-6, -0.5);
            for (uint32_t index = 0; index < k_dim; ++index) {
                queries[((size_t)row * k_heads + head) * k_dim + index] =
                    (float)((double)q_in[index] * q_inverse * query_scale);
                keys[((size_t)row * k_heads + head) * k_dim + index] =
                    (float)((double)k_in[index] * k_inverse * 1.0);
            }
        }
        memcpy(values + (size_t)row * value_dim, post + 2U * key_dim,
               value_dim * sizeof(float));
    }

    for (uint32_t row = 0; row < seq; ++row) {
        for (uint32_t head = 0; head < v_heads; ++head) {
            outputs->gate[(size_t)row * v_heads + head] = (float)(
                -exp((double)weights->a_log[head]) *
                softplus((double)a[(size_t)row * v_heads + head] + (double)weights->dt_bias[head]));
        }
    }

    for (uint32_t row = 0; row < seq; ++row) {
        for (uint32_t head = 0; head < v_heads; ++head) {
            const double decay = exp((double)outputs->gate[(size_t)row * v_heads + head]);
            const double beta = sigmoid((double)b[(size_t)row * v_heads + head]);
            const float *k_vec = keys + ((size_t)row * k_heads + head) * k_dim;
            const float *q_vec = queries + ((size_t)row * k_heads + head) * k_dim;
            const float *v_vec = values + ((size_t)row * v_heads + head) * v_dim;
            double *head_state = state + (size_t)head * k_dim * v_dim;

            for (uint32_t ki = 0; ki < k_dim; ++ki) {
                for (uint32_t vi = 0; vi < v_dim; ++vi) {
                    head_state[(size_t)ki * v_dim + vi] *= decay;
                }
            }
            for (uint32_t vi = 0; vi < v_dim; ++vi) {
                double kv_mem = 0.0;
                for (uint32_t ki = 0; ki < k_dim; ++ki) {
                    kv_mem += head_state[(size_t)ki * v_dim + vi] * k_vec[ki];
                }
                const double delta = ((double)v_vec[vi] - kv_mem) * beta;
                for (uint32_t ki = 0; ki < k_dim; ++ki) {
                    head_state[(size_t)ki * v_dim + vi] += (double)k_vec[ki] * delta;
                }
            }
            for (uint32_t vi = 0; vi < v_dim; ++vi) {
                double total = 0.0;
                for (uint32_t ki = 0; ki < k_dim; ++ki) {
                    total += head_state[(size_t)ki * v_dim + vi] * q_vec[ki];
                }
                outputs->core_out[((size_t)row * v_heads + head) * v_dim + vi] = (float)total;
            }
        }
    }
    for (uint32_t index = 0; index < v_heads * k_dim * v_dim; ++index) {
        outputs->state[index] = (float)state[index];
    }

    for (uint32_t row = 0; row < seq; ++row) {
        for (uint32_t head = 0; head < v_heads; ++head) {
            const size_t base = ((size_t)row * v_heads + head) * v_dim;
            double mean_squared = 0.0;
            for (uint32_t vi = 0; vi < v_dim; ++vi) {
                mean_squared += (double)outputs->core_out[base + vi] * outputs->core_out[base + vi];
            }
            mean_squared = mean_squared / v_dim + config->rms_epsilon;
            const double inverse_root = pow(mean_squared, -0.5);
            for (uint32_t vi = 0; vi < v_dim; ++vi) {
                outputs->gated[base + vi] = (float)(
                    (double)outputs->core_out[base + vi] * inverse_root *
                    (double)weights->gated_norm[vi] * silu((double)z[base + vi]));
            }
        }
    }

    linear_forward(outputs->gated, weights->out_proj, seq, value_dim, hidden, outputs->mixer);
    for (uint32_t index = 0; index < seq * hidden; ++index) {
        outputs->after_mixer[index] = (float)((double)input[index] + outputs->mixer[index]);
    }

    rms_norm_one_plus(outputs->after_mixer, weights->post_norm, seq, hidden,
                      config->rms_epsilon, mlp_normed);
    linear_forward(mlp_normed, weights->gate_proj, seq, hidden, intermediate, gate_buffer);
    linear_forward(mlp_normed, weights->up_proj, seq, hidden, intermediate, up_buffer);
    for (uint32_t index = 0; index < seq * intermediate; ++index) {
        gate_buffer[index] = (float)(silu((double)gate_buffer[index]) * up_buffer[index]);
    }
    linear_forward(gate_buffer, weights->down_proj, seq, intermediate, hidden, mlp_out);
    for (uint32_t index = 0; index < seq * hidden; ++index) {
        outputs->after_mlp[index] = (float)((double)outputs->after_mixer[index] + mlp_out[index]);
    }
    result = 0;

cleanup:
    free(mixed);
    free(z);
    free(b);
    free(a);
    free(queries);
    free(keys);
    free(values);
    free(state);
    free(mlp_normed);
    free(gate_buffer);
    free(up_buffer);
    free(mlp_out);
    return result;
}

static void apply_partial_rope(float *states, uint32_t rows, uint32_t heads,
                               uint32_t head_dim, uint32_t rotary_dim, double theta)
{
    const uint32_t half = rotary_dim / 2U;
    for (uint32_t position = 0; position < rows; ++position) {
        for (uint32_t head = 0; head < heads; ++head) {
            float *vector = states + ((size_t)position * heads + head) * head_dim;
            for (uint32_t index = 0; index < half; ++index) {
                const double inv_freq = 1.0 / pow(theta, (2.0 * index) / (double)rotary_dim);
                const double angle = (double)position * inv_freq;
                const double cosine = cos(angle);
                const double sine = sin(angle);
                const double first = vector[index];
                const double second = vector[index + half];
                vector[index] = (float)(first * cosine - second * sine);
                vector[index + half] = (float)(second * cosine + first * sine);
            }
        }
    }
}

int qwen35_attention_layer_forward(const struct qwen35_layer_config *config,
                                   const struct qwen35_attention_weights *weights,
                                   const float *input,
                                   const struct qwen35_attention_outputs *outputs)
{
    const uint32_t seq = config->sequence_length;
    const uint32_t hidden = config->hidden_size;
    const uint32_t heads = config->query_heads;
    const uint32_t kv_heads = config->kv_heads;
    const uint32_t head_dim = config->head_dim;
    const uint32_t query_size = heads * head_dim;
    const uint32_t kv_size = kv_heads * head_dim;
    const uint32_t intermediate = config->intermediate_size;
    const uint32_t groups = heads / kv_heads;
    int result = -1;

    float *q_and_gate = malloc((size_t)seq * 2U * query_size * sizeof(float));
    float *attention_gate = malloc((size_t)seq * query_size * sizeof(float));
    float *value = malloc((size_t)seq * kv_size * sizeof(float));
    double *scores = malloc((size_t)seq * sizeof(double));
    float *mlp_normed = malloc((size_t)seq * hidden * sizeof(float));
    float *gate_buffer = malloc((size_t)seq * intermediate * sizeof(float));
    float *up_buffer = malloc((size_t)seq * intermediate * sizeof(float));
    float *mlp_out = malloc((size_t)seq * hidden * sizeof(float));
    if (!q_and_gate || !attention_gate || !value || !scores || !mlp_normed ||
        !gate_buffer || !up_buffer || !mlp_out) {
        goto cleanup;
    }

    rms_norm_one_plus(input, weights->input_norm, seq, hidden, config->rms_epsilon,
                      outputs->normed);
    linear_forward(outputs->normed, weights->q_proj, seq, hidden, 2U * query_size, q_and_gate);
    linear_forward(outputs->normed, weights->k_proj, seq, hidden, kv_size, outputs->key);
    linear_forward(outputs->normed, weights->v_proj, seq, hidden, kv_size, value);

    for (uint32_t row = 0; row < seq; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            const float *source = q_and_gate + ((size_t)row * heads + head) * 2U * head_dim;
            memcpy(outputs->query + ((size_t)row * heads + head) * head_dim,
                   source, head_dim * sizeof(float));
            memcpy(attention_gate + ((size_t)row * heads + head) * head_dim,
                   source + head_dim, head_dim * sizeof(float));
        }
    }

    rms_norm_one_plus(outputs->query, weights->q_norm, seq * heads, head_dim,
                      config->rms_epsilon, outputs->query);
    rms_norm_one_plus(outputs->key, weights->k_norm, seq * kv_heads, head_dim,
                      config->rms_epsilon, outputs->key);
    apply_partial_rope(outputs->query, seq, heads, head_dim, config->rotary_dim,
                       (double)config->rope_theta);
    apply_partial_rope(outputs->key, seq, kv_heads, head_dim, config->rotary_dim,
                       (double)config->rope_theta);

    const double scale = pow((double)head_dim, -0.5);
    for (uint32_t position = 0; position < seq; ++position) {
        for (uint32_t head = 0; head < heads; ++head) {
            const uint32_t kv_head = head / groups;
            const float *query_vector = outputs->query + ((size_t)position * heads + head) * head_dim;
            double maximum = -INFINITY;
            for (uint32_t key_position = 0; key_position <= position; ++key_position) {
                const float *key_vector =
                    outputs->key + ((size_t)key_position * kv_heads + kv_head) * head_dim;
                double raw = 0.0;
                for (uint32_t index = 0; index < head_dim; ++index) {
                    raw += (double)query_vector[index] * key_vector[index];
                }
                raw *= scale;
                scores[key_position] = raw;
                maximum = raw > maximum ? raw : maximum;
            }
            double denominator = 0.0;
            for (uint32_t key_position = 0; key_position <= position; ++key_position) {
                denominator += exp(scores[key_position] - maximum);
            }
            for (uint32_t index = 0; index < head_dim; ++index) {
                double total = 0.0;
                for (uint32_t key_position = 0; key_position <= position; ++key_position) {
                    const double weight = exp(scores[key_position] - maximum) / denominator;
                    total += weight *
                             value[((size_t)key_position * kv_heads + kv_head) * head_dim + index];
                }
                outputs->attention[((size_t)position * heads + head) * head_dim + index] =
                    (float)total;
            }
        }
    }

    for (uint32_t index = 0; index < seq * query_size; ++index) {
        outputs->gated_attention[index] = (float)(
            (double)outputs->attention[index] * sigmoid((double)attention_gate[index]));
    }
    linear_forward(outputs->gated_attention, weights->o_proj, seq, query_size, hidden,
                   outputs->mixer);
    for (uint32_t index = 0; index < seq * hidden; ++index) {
        outputs->after_mixer[index] = (float)((double)input[index] + outputs->mixer[index]);
    }

    rms_norm_one_plus(outputs->after_mixer, weights->post_norm, seq, hidden,
                      config->rms_epsilon, mlp_normed);
    linear_forward(mlp_normed, weights->gate_proj, seq, hidden, intermediate, gate_buffer);
    linear_forward(mlp_normed, weights->up_proj, seq, hidden, intermediate, up_buffer);
    for (uint32_t index = 0; index < seq * intermediate; ++index) {
        gate_buffer[index] = (float)(silu((double)gate_buffer[index]) * up_buffer[index]);
    }
    linear_forward(gate_buffer, weights->down_proj, seq, intermediate, hidden, mlp_out);
    for (uint32_t index = 0; index < seq * hidden; ++index) {
        outputs->after_mlp[index] = (float)((double)outputs->after_mixer[index] + mlp_out[index]);
    }
    result = 0;

cleanup:
    free(q_and_gate);
    free(attention_gate);
    free(value);
    free(scores);
    free(mlp_normed);
    free(gate_buffer);
    free(up_buffer);
    free(mlp_out);
    return result;
}
