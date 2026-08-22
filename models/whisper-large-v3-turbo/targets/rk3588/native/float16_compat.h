#pragma once

// GCC's AArch64 C++ frontend exposes IEEE binary16 as __fp16, while Clang
// exposes the standard spelling _Float16. Both are two-byte storage types and
// map to RKNPU2's RKNN_TENSOR_FLOAT16 buffers.
#if defined(__GNUC__) && !defined(__clang__) && defined(__aarch64__)
using llmc_float16 = __fp16;
#else
using llmc_float16 = _Float16;
#endif

static_assert(sizeof(llmc_float16) == 2, "FP16 storage must be two bytes");
