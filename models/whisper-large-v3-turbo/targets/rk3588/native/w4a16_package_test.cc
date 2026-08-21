#include "w4a16_model.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

void require_shape(const llmc::W4Model &model, const std::string &name,
                   uint32_t rows, uint32_t columns, llmc::W4Encoding encoding) {
  const auto &tensor = model.require(name);
  if (tensor.dimensions != 2 || tensor.shape[0] != rows ||
      tensor.shape[1] != columns || tensor.encoding != encoding) {
    throw std::runtime_error("unexpected tensor definition: " + name);
  }
}

void require_vector(const llmc::W4Model &model, const std::string &name,
                    uint32_t elements) {
  const auto &tensor = model.require(name);
  if (tensor.dimensions != 1 || tensor.shape[0] != elements ||
      tensor.encoding != llmc::W4Encoding::kFloat16) {
    throw std::runtime_error("unexpected vector definition: " + name);
  }
}

void validate_tensor_payloads(const llmc::W4Model &model) {
  size_t quantized = 0;
  size_t fp16 = 0;
  for (const auto &pair : model.tensors()) {
    const auto &tensor = pair.second;
    if (tensor.encoding == llmc::W4Encoding::kW4A16Group) {
      ++quantized;
      if (tensor.shape[0] % 64 != 0 || tensor.shape[1] % 32 != 0) {
        throw std::runtime_error("RK3588 INT4 alignment failure: " +
                                 tensor.name);
      }
      const size_t scale_count = tensor.scale_size / sizeof(float);
      for (size_t index = 0; index < scale_count; ++index) {
        if (!std::isfinite(tensor.scales[index]) ||
            tensor.scales[index] <= 0.0f) {
          throw std::runtime_error("invalid quantization scale: " +
                                   tensor.name);
        }
      }
    } else {
      ++fp16;
    }
  }
  std::printf("quantized_tensors: %zu\n", quantized);
  std::printf("fp16_tensors: %zu\n", fp16);
}

void validate_encoder(const llmc::W4Model &model) {
  if (model.tensor_count() != 499) {
    throw std::runtime_error("encoder package must contain 499 tensors");
  }
  require_shape(model, "model.encoder.conv1.weight", 1280, 384,
                llmc::W4Encoding::kW4A16Group);
  require_shape(model, "model.encoder.conv2.weight", 1280, 3840,
                llmc::W4Encoding::kW4A16Group);
  require_shape(model, "model.encoder.embed_positions.weight", 1500, 1280,
                llmc::W4Encoding::kFloat16);
  require_vector(model, "model.encoder.conv1.bias", 1280);
  require_vector(model, "model.encoder.conv2.bias", 1280);
  require_vector(model, "model.encoder.layer_norm.weight", 1280);
  require_vector(model, "model.encoder.layer_norm.bias", 1280);
  for (int layer = 0; layer < 32; ++layer) {
    const std::string prefix =
        "model.encoder.layers." + std::to_string(layer) + ".";
    require_shape(model, prefix + "self_attn.q_proj.weight", 1280, 1280,
                  llmc::W4Encoding::kW4A16Group);
    require_shape(model, prefix + "self_attn.k_proj.weight", 1280, 1280,
                  llmc::W4Encoding::kW4A16Group);
    require_shape(model, prefix + "self_attn.v_proj.weight", 1280, 1280,
                  llmc::W4Encoding::kW4A16Group);
    require_shape(model, prefix + "self_attn.out_proj.weight", 1280, 1280,
                  llmc::W4Encoding::kW4A16Group);
    require_shape(model, prefix + "fc1.weight", 5120, 1280,
                  llmc::W4Encoding::kW4A16Group);
    require_shape(model, prefix + "fc2.weight", 1280, 5120,
                  llmc::W4Encoding::kW4A16Group);
    for (const char *name :
         {"self_attn.q_proj.bias", "self_attn.v_proj.bias",
          "self_attn.out_proj.bias", "self_attn_layer_norm.weight",
          "self_attn_layer_norm.bias", "final_layer_norm.weight",
          "final_layer_norm.bias", "fc2.bias"}) {
      require_vector(model, prefix + name, 1280);
    }
    require_vector(model, prefix + "fc1.bias", 5120);
  }
  for (int layer = 0; layer < 4; ++layer) {
    const std::string prefix =
        "model.decoder.layers." + std::to_string(layer) + ".encoder_attn.";
    require_shape(model, prefix + "k_proj.weight", 1280, 1280,
                  llmc::W4Encoding::kW4A16Group);
    require_shape(model, prefix + "v_proj.weight", 1280, 1280,
                  llmc::W4Encoding::kW4A16Group);
    require_vector(model, prefix + "v_proj.bias", 1280);
  }
}

void validate_decoder(const llmc::W4Model &model) {
  if (model.tensor_count() != 101) {
    throw std::runtime_error("decoder package must contain 101 tensors");
  }
  require_shape(model, "model.decoder.embed_tokens.weight", 51866, 1280,
                llmc::W4Encoding::kFloat16);
  require_shape(model, "model.decoder.embed_positions.weight", 448, 1280,
                llmc::W4Encoding::kFloat16);
  require_shape(model, "model.proj_out.weight", 51904, 1280,
                llmc::W4Encoding::kW4A16Group);
  require_vector(model, "model.decoder.layer_norm.weight", 1280);
  require_vector(model, "model.decoder.layer_norm.bias", 1280);
  for (int layer = 0; layer < 4; ++layer) {
    const std::string prefix =
        "model.decoder.layers." + std::to_string(layer) + ".";
    for (const char *attention : {"self_attn.", "encoder_attn."}) {
      const std::string base = prefix + attention;
      require_shape(model, base + "q_proj.weight", 1280, 1280,
                    llmc::W4Encoding::kW4A16Group);
      require_shape(model, base + "k_proj.weight", 1280, 1280,
                    llmc::W4Encoding::kW4A16Group);
      require_shape(model, base + "v_proj.weight", 1280, 1280,
                    llmc::W4Encoding::kW4A16Group);
      require_shape(model, base + "out_proj.weight", 1280, 1280,
                    llmc::W4Encoding::kW4A16Group);
      require_vector(model, base + "q_proj.bias", 1280);
      require_vector(model, base + "v_proj.bias", 1280);
      require_vector(model, base + "out_proj.bias", 1280);
    }
    require_shape(model, prefix + "fc1.weight", 5120, 1280,
                  llmc::W4Encoding::kW4A16Group);
    require_shape(model, prefix + "fc2.weight", 1280, 5120,
                  llmc::W4Encoding::kW4A16Group);
    for (const char *name :
         {"self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
          "encoder_attn_layer_norm.weight", "encoder_attn_layer_norm.bias",
          "final_layer_norm.weight", "final_layer_norm.bias", "fc2.bias"}) {
      require_vector(model, prefix + name, 1280);
    }
    require_vector(model, prefix + "fc1.bias", 5120);
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s ENCODER.llmc DECODER.llmc\n", argv[0]);
    return 2;
  }
  try {
    llmc::W4Model encoder(argv[1]);
    llmc::W4Model decoder(argv[2]);
    validate_encoder(encoder);
    validate_tensor_payloads(encoder);
    std::printf("encoder_bytes: %llu\n",
                static_cast<unsigned long long>(encoder.file_size()));
    validate_decoder(decoder);
    validate_tensor_payloads(decoder);
    std::printf("decoder_bytes: %llu\n",
                static_cast<unsigned long long>(decoder.file_size()));
    std::printf("package_validation: passed\n");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "package_validation: failed: %s\n", error.what());
    return 1;
  }
  return 0;
}
