#include "minimindo_layer.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int valid_config(const minimindo_layer_config *config)
{
    return config != NULL && config->sequence_length != 0U &&
           config->hidden_size != 0U && config->query_heads != 0U &&
           config->kv_heads != 0U && config->head_dim != 0U &&
           config->intermediate_size != 0U &&
           config->query_heads % config->kv_heads == 0U &&
           config->query_heads * config->head_dim == config->hidden_size &&
           (config->head_dim & 1U) == 0U && config->rms_epsilon > 0.0f &&
           config->rope_theta > 0.0f;
}

static void linear(const float *input, const float *weights, float *output,
                   uint32_t rows, uint32_t input_size, uint32_t output_size)
{
    for (uint32_t row = 0; row < rows; ++row) {
        const float *input_row = input + (size_t)row * input_size;
        float *output_row = output + (size_t)row * output_size;
        for (uint32_t out = 0; out < output_size; ++out) {
            const float *weight_row = weights + (size_t)out * input_size;
            double sum = 0.0;
            for (uint32_t in = 0; in < input_size; ++in)
                sum += (double)input_row[in] * (double)weight_row[in];
            output_row[out] = (float)sum;
        }
    }
}

static void rms_norm(const float *input, const float *weight, float *output,
                     uint32_t rows, uint32_t width, float epsilon)
{
    for (uint32_t row = 0; row < rows; ++row) {
        const float *input_row = input + (size_t)row * width;
        float *output_row = output + (size_t)row * width;
        double squares = 0.0;
        for (uint32_t index = 0; index < width; ++index)
            squares += (double)input_row[index] * input_row[index];
        const double scale = 1.0 / sqrt(squares / width + epsilon);
        for (uint32_t index = 0; index < width; ++index)
            output_row[index] =
                (float)((double)input_row[index] * scale * weight[index]);
    }
}

static void apply_rope(float *states, uint32_t rows, uint32_t heads,
                       uint32_t head_dim, uint32_t start_position, float theta)
{
    const uint32_t half = head_dim / 2U;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t position = start_position + row;
        for (uint32_t head = 0; head < heads; ++head) {
            float *vector = states + ((size_t)row * heads + head) * head_dim;
            for (uint32_t index = 0; index < half; ++index) {
                const double exponent = (2.0 * index) / head_dim;
                const double angle = position / pow((double)theta, exponent);
                const double cosine = cos(angle), sine = sin(angle);
                const double first = vector[index];
                const double second = vector[index + half];
                vector[index] = (float)(first * cosine - second * sine);
                vector[index + half] =
                    (float)(second * cosine + first * sine);
            }
        }
    }
}

static void causal_attention(const minimindo_layer_config *config,
                             const float *query, const float *key,
                             const float *value, double *scores, float *output)
{
    const uint32_t rows = config->sequence_length;
    const uint32_t heads = config->query_heads;
    const uint32_t kv_heads = config->kv_heads;
    const uint32_t head_dim = config->head_dim;
    const uint32_t groups = heads / kv_heads;
    const double inverse = 1.0 / sqrt((double)head_dim);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            const uint32_t kv_head = head / groups;
            const float *q = query + ((size_t)row * heads + head) * head_dim;
            double maximum = -INFINITY;
            for (uint32_t source = 0; source <= row; ++source) {
                const float *k = key +
                    ((size_t)source * kv_heads + kv_head) * head_dim;
                double score = 0.0;
                for (uint32_t index = 0; index < head_dim; ++index)
                    score += (double)q[index] * k[index];
                scores[source] = score * inverse;
                if (scores[source] > maximum) maximum = scores[source];
            }
            double denominator = 0.0;
            for (uint32_t source = 0; source <= row; ++source) {
                scores[source] = exp(scores[source] - maximum);
                denominator += scores[source];
            }
            float *out = output + ((size_t)row * heads + head) * head_dim;
            for (uint32_t index = 0; index < head_dim; ++index) {
                double sum = 0.0;
                for (uint32_t source = 0; source <= row; ++source) {
                    const float *v = value +
                        ((size_t)source * kv_heads + kv_head) * head_dim;
                    sum += scores[source] * v[index];
                }
                out[index] = (float)(sum / denominator);
            }
        }
    }
}

static double silu(double value)
{
    return value / (1.0 + exp(-value));
}

size_t minimindo_layer_workspace_floats(const minimindo_layer_config *config)
{
    if (!valid_config(config)) return 0U;
    const size_t rows = config->sequence_length;
    return rows * ((size_t)config->hidden_size +
                   2U * config->intermediate_size) +
           2U * config->sequence_length;
}

int minimindo_layer_forward(const minimindo_layer_config *config,
                            const minimindo_layer_weights *weights,
                            const float *input, float *workspace,
                            size_t workspace_floats,
                            const minimindo_layer_outputs *outputs)
{
    if (!valid_config(config) || weights == NULL || input == NULL ||
        workspace == NULL || outputs == NULL) return -1;
    if (workspace_floats < minimindo_layer_workspace_floats(config)) return -2;
    if (weights->input_norm == NULL || weights->q_proj == NULL ||
        weights->k_proj == NULL || weights->v_proj == NULL ||
        weights->q_norm == NULL || weights->k_norm == NULL ||
        weights->o_proj == NULL || weights->post_attention_norm == NULL ||
        weights->gate_proj == NULL || weights->up_proj == NULL ||
        weights->down_proj == NULL || outputs->normalized_input == NULL ||
        outputs->query == NULL || outputs->key == NULL ||
        outputs->value == NULL || outputs->attention == NULL ||
        outputs->after_attention == NULL || outputs->normalized_mlp == NULL ||
        outputs->output == NULL) return -3;

    const uint32_t rows = config->sequence_length;
    const uint32_t hidden = config->hidden_size;
    const uint32_t kv_size = config->kv_heads * config->head_dim;
    const uint32_t intermediate = config->intermediate_size;
    float *attention_heads = workspace;
    float *gate = attention_heads + (size_t)rows * hidden;
    float *up = gate + (size_t)rows * intermediate;
    double *scores = (double *)(up + (size_t)rows * intermediate);

    rms_norm(input, weights->input_norm, outputs->normalized_input,
             rows, hidden, config->rms_epsilon);
    linear(outputs->normalized_input, weights->q_proj, outputs->query,
           rows, hidden, hidden);
    linear(outputs->normalized_input, weights->k_proj, outputs->key,
           rows, hidden, kv_size);
    linear(outputs->normalized_input, weights->v_proj, outputs->value,
           rows, hidden, kv_size);
    rms_norm(outputs->query, weights->q_norm, outputs->query,
             rows * config->query_heads, config->head_dim,
             config->rms_epsilon);
    rms_norm(outputs->key, weights->k_norm, outputs->key,
             rows * config->kv_heads, config->head_dim,
             config->rms_epsilon);
    apply_rope(outputs->query, rows, config->query_heads, config->head_dim,
               config->start_position, config->rope_theta);
    apply_rope(outputs->key, rows, config->kv_heads, config->head_dim,
               config->start_position, config->rope_theta);
    causal_attention(config, outputs->query, outputs->key, outputs->value,
                     scores, attention_heads);
    linear(attention_heads, weights->o_proj, outputs->attention,
           rows, hidden, hidden);
    for (size_t index = 0; index < (size_t)rows * hidden; ++index)
        outputs->after_attention[index] = input[index] + outputs->attention[index];
    rms_norm(outputs->after_attention, weights->post_attention_norm,
             outputs->normalized_mlp, rows, hidden, config->rms_epsilon);
    linear(outputs->normalized_mlp, weights->gate_proj, gate,
           rows, hidden, intermediate);
    linear(outputs->normalized_mlp, weights->up_proj, up,
           rows, hidden, intermediate);
    for (size_t index = 0; index < (size_t)rows * intermediate; ++index)
        gate[index] = (float)(silu(gate[index]) * up[index]);
    linear(gate, weights->down_proj, outputs->output,
           rows, intermediate, hidden);
    for (size_t index = 0; index < (size_t)rows * hidden; ++index)
        outputs->output[index] += outputs->after_attention[index];
    return 0;
}
