#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rknn_api.h"

namespace {

const char* type_name(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32: return "float32";
    case RKNN_TENSOR_FLOAT16: return "float16";
    case RKNN_TENSOR_INT8: return "int8";
    case RKNN_TENSOR_UINT8: return "uint8";
    case RKNN_TENSOR_INT16: return "int16";
    case RKNN_TENSOR_UINT16: return "uint16";
    case RKNN_TENSOR_INT32: return "int32";
    case RKNN_TENSOR_UINT32: return "uint32";
    case RKNN_TENSOR_INT64: return "int64";
    case RKNN_TENSOR_BOOL: return "bool";
    default: return "unknown";
  }
}

void print_attr(const char* direction, rknn_tensor_attr* attr) {
  std::printf("%s[%u] name=%s type=%s fmt=%d qnt=%d dims=[",
              direction, attr->index, attr->name, type_name(attr->type),
              static_cast<int>(attr->fmt), static_cast<int>(attr->qnt_type));
  for (uint32_t i = 0; i < attr->n_dims; ++i) {
    std::printf("%s%u", i == 0 ? "" : ",", attr->dims[i]);
  }
  std::printf("] elems=%u size=%u size_stride=%u w_stride=%u h_stride=%u\n",
              attr->n_elems, attr->size, attr->size_with_stride,
              attr->w_stride, attr->h_stride);
}

int query(const char* path) {
  rknn_context ctx = 0;
  int rc = rknn_init(&ctx, const_cast<char*>(path), 0, 0, nullptr);
  if (rc != RKNN_SUCC) {
    std::fprintf(stderr, "rknn_init(%s) failed: %d\n", path, rc);
    return 1;
  }

  rknn_sdk_version version{};
  if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == RKNN_SUCC) {
    std::printf("model=%s api=%s driver=%s\n", path, version.api_version,
                version.drv_version);
  }

  rknn_input_output_num count{};
  rc = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count));
  if (rc != RKNN_SUCC) {
    std::fprintf(stderr, "RKNN_QUERY_IN_OUT_NUM failed: %d\n", rc);
    rknn_destroy(ctx);
    return 1;
  }
  std::printf("inputs=%u outputs=%u\n", count.n_input, count.n_output);
  for (uint32_t i = 0; i < count.n_input; ++i) {
    rknn_tensor_attr attr{};
    attr.index = i;
    rc = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (rc != RKNN_SUCC) {
      std::fprintf(stderr, "input attr %u failed: %d\n", i, rc);
      rknn_destroy(ctx);
      return 1;
    }
    print_attr("input", &attr);
  }
  for (uint32_t i = 0; i < count.n_output; ++i) {
    rknn_tensor_attr attr{};
    attr.index = i;
    rc = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
    if (rc != RKNN_SUCC) {
      std::fprintf(stderr, "output attr %u failed: %d\n", i, rc);
      rknn_destroy(ctx);
      return 1;
    }
    print_attr("output", &attr);
  }
  rknn_destroy(ctx);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s MODEL.rknn [MODEL.rknn ...]\n", argv[0]);
    return 2;
  }
  int status = 0;
  for (int i = 1; i < argc; ++i) {
    status |= query(argv[i]);
  }
  return status;
}
