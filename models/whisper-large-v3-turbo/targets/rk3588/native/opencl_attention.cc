#define CL_TARGET_OPENCL_VERSION 120

#include "opencl_attention.h"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace llmc {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kHeads = 20;
constexpr size_t kHead = 64;
constexpr size_t kModel = kHeads * kHead;
constexpr size_t kMaxRows = 1500;
constexpr size_t kMaxKeyStride = 1504;

constexpr const char *kSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// A 16x16 workgroup computes one query/key tile for one attention head. Q and
// K are loaded once into local memory and reused for 256 dot products.
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void attention_qk(
    __global const half *restrict query,
    __global const half *restrict key,
    __global half *restrict scores,
    const uint query_rows,
    const uint key_rows,
    const uint score_columns,
    const uint key_stride,
    const uint packed_key_value) {
  const uint key_tile = get_group_id(0);
  const uint query_tile = get_group_id(1);
  const uint head = get_group_id(2);
  const uint key_lane = get_local_id(0);
  const uint query_lane = get_local_id(1);
  const uint linear = query_lane * 16u + key_lane;
  __local half query_tile_data[16][64];
  __local half key_tile_data[16][64];

#pragma unroll
  for (uint load = 0; load < 4u; ++load) {
    const uint element = linear + load * 256u;
    const uint tile_row = element / 64u;
    const uint dimension = element & 63u;
    const uint query_row = query_tile * 16u + tile_row;
    const uint key_row = key_tile * 16u + tile_row;
    query_tile_data[tile_row][dimension] =
        query_row < query_rows
            ? query[(size_t)query_row * 1280u + head * 64u + dimension]
            : convert_half(0.0f);
    half key_value = convert_half(0.0f);
    if (key_row < key_rows) {
      key_value = packed_key_value
                      ? key[(size_t)head * 64u * key_stride +
                            (size_t)dimension * key_stride + key_row]
                      : key[(size_t)key_row * 1280u + head * 64u + dimension];
    }
    key_tile_data[tile_row][dimension] = key_value;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint query_row = query_tile * 16u + query_lane;
  const uint key_row = key_tile * 16u + key_lane;
  if (query_row >= query_rows || key_row >= key_rows)
    return;
  float sum = 0.0f;
#pragma unroll
  for (uint dimension = 0; dimension < 64u; dimension += 4u) {
    const float4 q = convert_float4(
        vload4(0, query_tile_data[query_lane] + dimension));
    const float4 k =
        convert_float4(vload4(0, key_tile_data[key_lane] + dimension));
    sum += dot(q, k);
  }
  const size_t offset =
      ((size_t)head * query_rows + query_row) * score_columns + key_row;
  scores[offset] = convert_half_rte(sum * 0.125f);
}

// One 256-thread workgroup normalizes one complete attention row in place.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void attention_softmax(
    __global half *scores,
    const uint query_rows,
    const uint key_rows,
    const uint score_columns) {
  const uint query_row = get_group_id(0);
  const uint head = get_group_id(1);
  const uint lane = get_local_id(0);
  __local float reduction[256];
  const size_t base =
      ((size_t)head * query_rows + query_row) * score_columns;

  float maximum = -3.402823466e+38f;
  for (uint column = lane; column < key_rows; column += 256u)
    maximum = fmax(maximum, convert_float(scores[base + column]));
  reduction[lane] = maximum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (lane < stride)
      reduction[lane] = fmax(reduction[lane], reduction[lane + stride]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  maximum = reduction[0];

  float total = 0.0f;
  for (uint column = lane; column < key_rows; column += 256u)
    total += exp(convert_float(scores[base + column]) - maximum);
  reduction[lane] = total;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (lane < stride)
      reduction[lane] += reduction[lane + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inverse = 1.0f / reduction[0];
  for (uint column = lane; column < score_columns; column += 256u) {
    scores[base + column] =
        column < key_rows
            ? convert_half_rte(
                  exp(convert_float(scores[base + column]) - maximum) *
                  inverse)
            : convert_half(0.0f);
  }
}

// One 64-thread workgroup computes one query row/head output. Probability is
// broadcast across all 64 dimensions while V reads remain contiguous.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_pv(
    __global const half *restrict probabilities,
    __global const half *restrict value,
    __global half *restrict output,
    const uint query_rows,
    const uint key_rows,
    const uint score_columns,
    const uint key_stride,
    const uint packed_key_value) {
  const uint dimension = get_local_id(0);
  const uint query_row = get_group_id(1);
  const uint head = get_group_id(2);
  const size_t probability_base =
      ((size_t)head * query_rows + query_row) * score_columns;
  float sum = 0.0f;
  for (uint key_row = 0; key_row < key_rows; ++key_row) {
    const float probability =
        convert_float(probabilities[probability_base + key_row]);
    const half value_half =
        packed_key_value
            ? value[(size_t)head * key_stride * 64u +
                    (size_t)key_row * 64u + dimension]
            : value[(size_t)key_row * 1280u + head * 64u + dimension];
    sum = fma(probability, convert_float(value_half), sum);
  }
  output[(size_t)query_row * 1280u + head * 64u + dimension] =
      convert_half_rte(sum);
}
)CLC";

[[noreturn]] void fail(const char *operation, cl_int status) {
  throw std::runtime_error(std::string(operation) + " failed: OpenCL " +
                           std::to_string(status));
}

void check(cl_int status, const char *operation) {
  if (status != CL_SUCCESS)
    fail(operation, status);
}

double event_ms(cl_event event) {
  cl_ulong start = 0;
  cl_ulong end = 0;
  check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(start), &start, nullptr),
        "clGetEventProfilingInfo(start)");
  check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end),
                                &end, nullptr),
        "clGetEventProfilingInfo(end)");
  return static_cast<double>(end - start) * 1.0e-6;
}

size_t bytes(size_t first, size_t second) {
  if (first == 0 || second == 0 ||
      first > std::numeric_limits<size_t>::max() / second ||
      first * second > std::numeric_limits<size_t>::max() /
                           sizeof(llmc_float16)) {
    throw std::runtime_error("OpenCL attention size overflow");
  }
  return first * second * sizeof(llmc_float16);
}

} // namespace

OpenClAttention::OpenClAttention(OpenClW4Context &context)
    : context_(context) {
  cl_int status = CL_SUCCESS;
  const char *source = kSource;
  const size_t source_size = std::char_traits<char>::length(source);
  program_ = clCreateProgramWithSource(context_.context(), 1, &source,
                                       &source_size, &status);
  check(status, "clCreateProgramWithSource(attention)");
  const cl_device_id device = context_.device();
  status = clBuildProgram(program_, 1, &device,
                          "-cl-fast-relaxed-math", nullptr, nullptr);
  if (status != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program_, context_.device(), CL_PROGRAM_BUILD_LOG, 0,
                          nullptr, &log_size);
    std::vector<char> log(log_size + 1, '\0');
    clGetProgramBuildInfo(program_, context_.device(), CL_PROGRAM_BUILD_LOG,
                          log_size, log.data(), nullptr);
    throw std::runtime_error("OpenCL attention build failed: " +
                             std::string(log.data()));
  }
  qk_kernel_ = clCreateKernel(program_, "attention_qk", &status);
  check(status, "clCreateKernel(attention_qk)");
  softmax_kernel_ = clCreateKernel(program_, "attention_softmax", &status);
  check(status, "clCreateKernel(attention_softmax)");
  pv_kernel_ = clCreateKernel(program_, "attention_pv", &status);
  check(status, "clCreateKernel(attention_pv)");

  query_ = clCreateBuffer(context_.context(), CL_MEM_READ_ONLY,
                          bytes(kMaxRows, kModel), nullptr, &status);
  check(status, "clCreateBuffer(attention query)");
  key_ = clCreateBuffer(context_.context(), CL_MEM_READ_ONLY,
                        bytes(kMaxKeyStride, kModel), nullptr, &status);
  check(status, "clCreateBuffer(attention key)");
  value_ = clCreateBuffer(context_.context(), CL_MEM_READ_ONLY,
                          bytes(kMaxKeyStride, kModel), nullptr, &status);
  check(status, "clCreateBuffer(attention value)");
  scores_ = clCreateBuffer(
      context_.context(), CL_MEM_READ_WRITE,
      bytes(kHeads * kMaxRows, kMaxKeyStride), nullptr, &status);
  check(status, "clCreateBuffer(attention scores)");
  output_ = clCreateBuffer(context_.context(), CL_MEM_WRITE_ONLY,
                           bytes(kMaxRows, kModel), nullptr, &status);
  check(status, "clCreateBuffer(attention output)");
}

OpenClAttention::~OpenClAttention() {
  if (output_ != nullptr)
    clReleaseMemObject(output_);
  if (scores_ != nullptr)
    clReleaseMemObject(scores_);
  if (value_ != nullptr)
    clReleaseMemObject(value_);
  if (key_ != nullptr)
    clReleaseMemObject(key_);
  if (query_ != nullptr)
    clReleaseMemObject(query_);
  if (pv_kernel_ != nullptr)
    clReleaseKernel(pv_kernel_);
  if (softmax_kernel_ != nullptr)
    clReleaseKernel(softmax_kernel_);
  if (qk_kernel_ != nullptr)
    clReleaseKernel(qk_kernel_);
  if (program_ != nullptr)
    clReleaseProgram(program_);
}

void OpenClAttention::run(const llmc_float16 *query,
                          const llmc_float16 *key,
                          const llmc_float16 *value,
                          llmc_float16 *output, size_t query_rows,
                          size_t key_rows, size_t key_stride,
                          bool packed_key_value) {
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || query_rows == 0 || query_rows > kMaxRows ||
      key_rows == 0 || key_rows > kMaxRows || key_stride < key_rows ||
      key_stride > kMaxKeyStride) {
    throw std::runtime_error("invalid OpenCL attention arguments");
  }
  const size_t score_columns = (key_rows + 15) / 16 * 16;
  const size_t query_bytes = bytes(query_rows, kModel);
  const size_t key_elements =
      (packed_key_value ? key_stride : key_rows) * kModel;
  const size_t key_bytes = bytes(key_elements, 1);
  const size_t output_bytes = bytes(query_rows, kModel);
  const auto wall_start = Clock::now();

  cl_event writes[3]{};
  cl_event qk_event = nullptr;
  cl_event softmax_event = nullptr;
  cl_event pv_event = nullptr;
  cl_event read_event = nullptr;
  check(clEnqueueWriteBuffer(context_.queue(), query_, CL_FALSE, 0,
                             query_bytes, query, 0, nullptr, &writes[0]),
        "clEnqueueWriteBuffer(attention query)");
  check(clEnqueueWriteBuffer(context_.queue(), key_, CL_FALSE, 0, key_bytes,
                             key, 0, nullptr, &writes[1]),
        "clEnqueueWriteBuffer(attention key)");
  check(clEnqueueWriteBuffer(context_.queue(), value_, CL_FALSE, 0, key_bytes,
                             value, 0, nullptr, &writes[2]),
        "clEnqueueWriteBuffer(attention value)");

  const cl_uint query_count = static_cast<cl_uint>(query_rows);
  const cl_uint key_count = static_cast<cl_uint>(key_rows);
  const cl_uint score_count = static_cast<cl_uint>(score_columns);
  const cl_uint key_storage_stride = static_cast<cl_uint>(key_stride);
  const cl_uint packed = packed_key_value ? 1u : 0u;
  check(clSetKernelArg(qk_kernel_, 0, sizeof(query_), &query_),
        "clSetKernelArg(qk query)");
  check(clSetKernelArg(qk_kernel_, 1, sizeof(key_), &key_),
        "clSetKernelArg(qk key)");
  check(clSetKernelArg(qk_kernel_, 2, sizeof(scores_), &scores_),
        "clSetKernelArg(qk scores)");
  check(clSetKernelArg(qk_kernel_, 3, sizeof(query_count), &query_count),
        "clSetKernelArg(qk query rows)");
  check(clSetKernelArg(qk_kernel_, 4, sizeof(key_count), &key_count),
        "clSetKernelArg(qk key rows)");
  check(clSetKernelArg(qk_kernel_, 5, sizeof(score_count), &score_count),
        "clSetKernelArg(qk score columns)");
  check(clSetKernelArg(qk_kernel_, 6, sizeof(key_storage_stride),
                       &key_storage_stride),
        "clSetKernelArg(qk key stride)");
  check(clSetKernelArg(qk_kernel_, 7, sizeof(packed), &packed),
        "clSetKernelArg(qk packed)");
  const size_t qk_local[3] = {16, 16, 1};
  const size_t qk_global[3] = {score_columns,
                               (query_rows + 15) / 16 * 16, kHeads};
  check(clEnqueueNDRangeKernel(context_.queue(), qk_kernel_, 3, nullptr,
                               qk_global, qk_local, 3, writes, &qk_event),
        "clEnqueueNDRangeKernel(attention qk)");

  check(clSetKernelArg(softmax_kernel_, 0, sizeof(scores_), &scores_),
        "clSetKernelArg(softmax scores)");
  check(clSetKernelArg(softmax_kernel_, 1, sizeof(query_count), &query_count),
        "clSetKernelArg(softmax query rows)");
  check(clSetKernelArg(softmax_kernel_, 2, sizeof(key_count), &key_count),
        "clSetKernelArg(softmax key rows)");
  check(clSetKernelArg(softmax_kernel_, 3, sizeof(score_count), &score_count),
        "clSetKernelArg(softmax score columns)");
  const size_t softmax_local[2] = {256, 1};
  const size_t softmax_global[2] = {query_rows * 256, kHeads};
  check(clEnqueueNDRangeKernel(context_.queue(), softmax_kernel_, 2, nullptr,
                               softmax_global, softmax_local, 1, &qk_event,
                               &softmax_event),
        "clEnqueueNDRangeKernel(attention softmax)");

  check(clSetKernelArg(pv_kernel_, 0, sizeof(scores_), &scores_),
        "clSetKernelArg(pv probabilities)");
  check(clSetKernelArg(pv_kernel_, 1, sizeof(value_), &value_),
        "clSetKernelArg(pv value)");
  check(clSetKernelArg(pv_kernel_, 2, sizeof(output_), &output_),
        "clSetKernelArg(pv output)");
  check(clSetKernelArg(pv_kernel_, 3, sizeof(query_count), &query_count),
        "clSetKernelArg(pv query rows)");
  check(clSetKernelArg(pv_kernel_, 4, sizeof(key_count), &key_count),
        "clSetKernelArg(pv key rows)");
  check(clSetKernelArg(pv_kernel_, 5, sizeof(score_count), &score_count),
        "clSetKernelArg(pv score columns)");
  check(clSetKernelArg(pv_kernel_, 6, sizeof(key_storage_stride),
                       &key_storage_stride),
        "clSetKernelArg(pv key stride)");
  check(clSetKernelArg(pv_kernel_, 7, sizeof(packed), &packed),
        "clSetKernelArg(pv packed)");
  const size_t pv_local[3] = {64, 1, 1};
  const size_t pv_global[3] = {64, query_rows, kHeads};
  check(clEnqueueNDRangeKernel(context_.queue(), pv_kernel_, 3, nullptr,
                               pv_global, pv_local, 1, &softmax_event,
                               &pv_event),
        "clEnqueueNDRangeKernel(attention pv)");
  check(clEnqueueReadBuffer(context_.queue(), output_, CL_FALSE, 0,
                            output_bytes, output, 1, &pv_event, &read_event),
        "clEnqueueReadBuffer(attention output)");
  check(clWaitForEvents(1, &read_event), "clWaitForEvents(attention read)");

  metrics_.blocks += 1;
  metrics_.qk_ms += event_ms(qk_event);
  metrics_.softmax_ms += event_ms(softmax_event);
  metrics_.pv_ms += event_ms(pv_event);
  metrics_.transfer_ms += event_ms(writes[0]) + event_ms(writes[1]) +
                          event_ms(writes[2]) + event_ms(read_event);
  metrics_.wall_ms += std::chrono::duration<double, std::milli>(
                          Clock::now() - wall_start)
                          .count();
  clReleaseEvent(read_event);
  clReleaseEvent(pv_event);
  clReleaseEvent(softmax_event);
  clReleaseEvent(qk_event);
  for (cl_event event : writes)
    clReleaseEvent(event);
}

} // namespace llmc
