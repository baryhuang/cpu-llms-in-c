#include "cpu_llms/gemma4_layer.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static void linear(const float *input, const float *weights, float *output,
                   size_t rows, size_t input_size, size_t output_size)
{
    for (size_t row = 0; row < rows; ++row) {
        const float *input_row = input + row * input_size;
        float *output_row = output + row * output_size;

        for (size_t output_index = 0; output_index < output_size; ++output_index) {
            const float *weight_row = weights + output_index * input_size;
            double sum = 0.0;

            for (size_t input_index = 0; input_index < input_size; ++input_index) {
                sum += (double)input_row[input_index] * (double)weight_row[input_index];
            }
            output_row[output_index] = (float)sum;
        }
    }
}

static void rms_norm(const float *input, const float *scale, float *output,
                     size_t rows, size_t width, float epsilon)
{
    for (size_t row = 0; row < rows; ++row) {
        const float *input_row = input + row * width;
        float *output_row = output + row * width;
        double squares = 0.0;

        for (size_t index = 0; index < width; ++index) {
            squares += (double)input_row[index] * (double)input_row[index];
        }

        const double inverse_root = pow(squares / (double)width + (double)epsilon, -0.5);
        for (size_t index = 0; index < width; ++index) {
            const double scaled = (double)input_row[index] * inverse_root;
            output_row[index] = (float)(scale == NULL ? scaled : scaled * (double)scale[index]);
        }
    }
}

static float gelu_pytorch_tanh(float value)
{
    const double x = value;
    const double coefficient = 0.79788456080286535588;
    const double inner = coefficient * (x + 0.044715 * x * x * x);
    return (float)(0.5 * x * (1.0 + tanh(inner)));
}

static void apply_rope(float *states, size_t sequence_length, size_t heads,
                       size_t head_dim, float theta)
{
    const size_t half = head_dim / 2U;

    for (size_t position = 0; position < sequence_length; ++position) {
        for (size_t head = 0; head < heads; ++head) {
            float *vector = states + (position * heads + head) * head_dim;

            for (size_t index = 0; index < half; ++index) {
                const double exponent = (double)(2U * index) / (double)head_dim;
                const double inverse_frequency = 1.0 / pow((double)theta, exponent);
                const double angle = (double)position * inverse_frequency;
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

static void attention_mqa(const struct cllm_gemma4_layer_config *config,
                          const float *query, const float *key, const float *value,
                          size_t sequence_length, float *scores, float *output)
{
    const size_t query_heads = config->query_heads;
    const size_t kv_heads = config->kv_heads;
    const size_t head_dim = config->head_dim;
    const size_t groups = query_heads / kv_heads;

    for (size_t query_position = 0; query_position < sequence_length; ++query_position) {
        size_t first_key = 0;
        if (config->sliding_window != 0U && query_position + 1U > config->sliding_window) {
            first_key = query_position + 1U - config->sliding_window;
        }

        for (size_t query_head = 0; query_head < query_heads; ++query_head) {
            const size_t kv_head = query_head / groups;
            const float *query_vector = query + (query_position * query_heads + query_head) * head_dim;
            double maximum = -INFINITY;
            double denominator = 0.0;

            for (size_t key_position = first_key; key_position <= query_position; ++key_position) {
                const float *key_vector = key + (key_position * kv_heads + kv_head) * head_dim;
                double score = 0.0;

                for (size_t index = 0; index < head_dim; ++index) {
                    score += (double)query_vector[index] * (double)key_vector[index];
                }
                scores[key_position] = (float)score;
                if (score > maximum) {
                    maximum = score;
                }
            }

            for (size_t key_position = first_key; key_position <= query_position; ++key_position) {
                const double exponential = exp((double)scores[key_position] - maximum);
                scores[key_position] = (float)exponential;
                denominator += exponential;
            }

            for (size_t index = 0; index < head_dim; ++index) {
                double sum = 0.0;
                for (size_t key_position = first_key; key_position <= query_position; ++key_position) {
                    const float *value_vector = value + (key_position * kv_heads + kv_head) * head_dim;
                    sum += ((double)scores[key_position] / denominator) * (double)value_vector[index];
                }
                output[(query_position * query_heads + query_head) * head_dim + index] = (float)sum;
            }
        }
    }
}

static int valid_config(const struct cllm_gemma4_layer_config *config, size_t sequence_length)
{
    if (config == NULL || sequence_length == 0U || config->hidden_size == 0U ||
        config->query_heads == 0U || config->kv_heads == 0U || config->head_dim == 0U ||
        config->intermediate_size == 0U || config->ple_size == 0U ||
        config->query_heads % config->kv_heads != 0U || config->head_dim % 2U != 0U ||
        config->query_heads * config->head_dim == 0U || config->rms_epsilon <= 0.0f ||
        config->rope_theta <= 0.0f) {
        return 0;
    }
    return 1;
}

size_t cllm_gemma4_layer_workspace_floats(const struct cllm_gemma4_layer_config *config,
                                          size_t sequence_length)
{
    if (!valid_config(config, sequence_length)) {
        return 0U;
    }

    return sequence_length * ((size_t)config->hidden_size +
                              2U * (size_t)config->intermediate_size +
                              (size_t)config->ple_size +
                              (size_t)config->query_heads * (size_t)config->head_dim) +
           sequence_length;
}

int cllm_gemma4_layer_forward(const struct cllm_gemma4_layer_config *config,
                              const struct cllm_gemma4_layer_weights *weights,
                              const float *input,
                              const float *per_layer_input,
                              size_t sequence_length,
                              float *workspace,
                              size_t workspace_floats,
                              const struct cllm_gemma4_layer_outputs *outputs)
{
    size_t required;
    size_t hidden_size;
    size_t query_size;
    size_t kv_size;
    size_t intermediate_size;
    size_t ple_size;
    float *normalized;
    float *gate;
    float *up;
    float *ple_gate;
    float *attention_heads;
    float *scores;

    if (!valid_config(config, sequence_length) || weights == NULL || input == NULL ||
        per_layer_input == NULL || workspace == NULL || outputs == NULL) {
        return -1;
    }

    required = cllm_gemma4_layer_workspace_floats(config, sequence_length);
    if (workspace_floats < required) {
        return -2;
    }

    if (weights->input_norm == NULL || weights->q_proj == NULL || weights->k_proj == NULL ||
        weights->v_proj == NULL || weights->q_norm == NULL || weights->k_norm == NULL ||
        weights->o_proj == NULL || weights->post_attention_norm == NULL ||
        weights->pre_feedforward_norm == NULL || weights->gate_proj == NULL ||
        weights->up_proj == NULL || weights->down_proj == NULL ||
        weights->post_feedforward_norm == NULL || weights->ple_gate == NULL ||
        weights->ple_projection == NULL || weights->post_ple_norm == NULL ||
        outputs->normalized_input == NULL || outputs->query == NULL || outputs->key == NULL ||
        outputs->value == NULL || outputs->attention == NULL ||
        outputs->after_attention == NULL || outputs->mlp == NULL || outputs->after_mlp == NULL ||
        outputs->ple == NULL || outputs->output == NULL) {
        return -3;
    }

    hidden_size = config->hidden_size;
    query_size = (size_t)config->query_heads * config->head_dim;
    kv_size = (size_t)config->kv_heads * config->head_dim;
    intermediate_size = config->intermediate_size;
    ple_size = config->ple_size;

    normalized = workspace;
    gate = normalized + sequence_length * hidden_size;
    up = gate + sequence_length * intermediate_size;
    ple_gate = up + sequence_length * intermediate_size;
    attention_heads = ple_gate + sequence_length * ple_size;
    scores = attention_heads + sequence_length * query_size;

    rms_norm(input, weights->input_norm, outputs->normalized_input,
             sequence_length, hidden_size, config->rms_epsilon);

    linear(outputs->normalized_input, weights->q_proj, outputs->query,
           sequence_length, hidden_size, query_size);
    linear(outputs->normalized_input, weights->k_proj, outputs->key,
           sequence_length, hidden_size, kv_size);
    linear(outputs->normalized_input, weights->v_proj, outputs->value,
           sequence_length, hidden_size, kv_size);

    rms_norm(outputs->query, weights->q_norm, outputs->query,
             sequence_length * config->query_heads, config->head_dim, config->rms_epsilon);
    rms_norm(outputs->key, weights->k_norm, outputs->key,
             sequence_length * config->kv_heads, config->head_dim, config->rms_epsilon);
    rms_norm(outputs->value, NULL, outputs->value,
             sequence_length * config->kv_heads, config->head_dim, config->rms_epsilon);

    apply_rope(outputs->query, sequence_length, config->query_heads, config->head_dim,
               config->rope_theta);
    apply_rope(outputs->key, sequence_length, config->kv_heads, config->head_dim,
               config->rope_theta);

    attention_mqa(config, outputs->query, outputs->key, outputs->value,
                  sequence_length, scores, attention_heads);
    linear(attention_heads, weights->o_proj, outputs->attention,
           sequence_length, query_size, hidden_size);
    rms_norm(outputs->attention, weights->post_attention_norm, normalized,
             sequence_length, hidden_size, config->rms_epsilon);
    for (size_t index = 0; index < sequence_length * hidden_size; ++index) {
        outputs->after_attention[index] = input[index] + normalized[index];
    }

    rms_norm(outputs->after_attention, weights->pre_feedforward_norm, normalized,
             sequence_length, hidden_size, config->rms_epsilon);
    linear(normalized, weights->gate_proj, gate,
           sequence_length, hidden_size, intermediate_size);
    linear(normalized, weights->up_proj, up,
           sequence_length, hidden_size, intermediate_size);
    for (size_t index = 0; index < sequence_length * intermediate_size; ++index) {
        gate[index] = gelu_pytorch_tanh(gate[index]) * up[index];
    }
    linear(gate, weights->down_proj, outputs->mlp,
           sequence_length, intermediate_size, hidden_size);
    rms_norm(outputs->mlp, weights->post_feedforward_norm, normalized,
             sequence_length, hidden_size, config->rms_epsilon);
    for (size_t index = 0; index < sequence_length * hidden_size; ++index) {
        outputs->after_mlp[index] = outputs->after_attention[index] + normalized[index];
    }

    linear(outputs->after_mlp, weights->ple_gate, ple_gate,
           sequence_length, hidden_size, ple_size);
    for (size_t index = 0; index < sequence_length * ple_size; ++index) {
        ple_gate[index] = gelu_pytorch_tanh(ple_gate[index]) * per_layer_input[index];
    }
    linear(ple_gate, weights->ple_projection, outputs->ple,
           sequence_length, ple_size, hidden_size);
    rms_norm(outputs->ple, weights->post_ple_norm, normalized,
             sequence_length, hidden_size, config->rms_epsilon);
    for (size_t index = 0; index < sequence_length * hidden_size; ++index) {
        outputs->output[index] =
            (outputs->after_mlp[index] + normalized[index]) * config->layer_scalar;
    }

    return 0;
}
