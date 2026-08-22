#define CL_TARGET_OPENCL_VERSION 120

#include "opencl_w4a16.h"

#include <chrono>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace llmc {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kTileColumns = 32;
constexpr size_t kTileInner = 32;
constexpr size_t kLocalInnerSplits = 4;

constexpr const char *kW4KernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// One 32x8 workgroup computes one row by 32 output columns. Work-items with
// the same local y consume adjacent packed nibbles, while the eight y lanes
// split K and reduce in local memory. Accumulation remains FP32.
__attribute__((reqd_work_group_size(32, 4, 1)))
__kernel void w4a16_linear_n32k32(
    __global const half *restrict input,
    __global const uchar *restrict weight,
    __global half *restrict output,
    const uint rows,
    const uint inner,
    const uint columns) {
  const uint column_tile = get_group_id(0);
  const uint row = get_group_id(1);
  const uint column_lane = get_local_id(0);
  const uint split = get_local_id(1);
  const uint column = column_tile * 32u + column_lane;
  const uint inner_tiles = inner / 32u;
  __local float partial[4][32];

  float accumulator = 0.0f;
  for (uint inner_tile = 0; inner_tile < inner_tiles; ++inner_tile) {
    const size_t record_index =
        (size_t)column_tile * (size_t)inner_tiles + inner_tile;
    __global const uchar *record = weight + record_index * 640u;
    __global const float *scales = (__global const float *)record;
    __global const uchar *packed = record + 128u;
    const float scale = scales[column_lane];

    for (uint inner_lane = split; inner_lane < 32u; inner_lane += 4u) {
      const uint packed_index = inner_lane * 32u + column_lane;
      const uchar byte = packed[packed_index >> 1];
      const uint nibble =
          (packed_index & 1u) ? ((uint)byte >> 4) : ((uint)byte & 15u);
      const int quantized = nibble < 8u ? (int)nibble : (int)nibble - 16;
      const uint k = inner_tile * 32u + inner_lane;
      accumulator += convert_float(input[(size_t)row * inner + k]) *
                     ((float)quantized * scale);
    }
  }

  partial[split][column_lane] = accumulator;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (split < 2u)
    partial[split][column_lane] += partial[split + 2u][column_lane];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (split == 0u) {
    const float sum = partial[0][column_lane] + partial[1][column_lane];
    if (row < rows && column < columns)
      output[(size_t)row * columns + column] = convert_half_rte(sum);
  }
}

// Decoder-specialized path. One work-item computes two adjacent output
// columns, so each packed byte and activation load feeds two FMAs. There are
// no workgroup barriers or local-memory reductions.
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void w4a16_linear_pair_n32k32(
    __global const half *restrict input,
    __global const uchar *restrict weight,
    __global half *restrict output,
    const uint rows,
    const uint inner,
    const uint columns) {
  const uint pair = get_global_id(0);
  const uint row = get_global_id(1);
  const uint column = pair * 2u;
  if (row >= rows || column >= columns)
    return;

  const uint column_tile = column / 32u;
  const uint column_lane = column & 31u;
  const uint inner_tiles = inner / 32u;
  float2 accumulator = (float2)(0.0f);
  for (uint inner_tile = 0; inner_tile < inner_tiles; ++inner_tile) {
    const size_t record_index =
        (size_t)column_tile * (size_t)inner_tiles + inner_tile;
    __global const uchar *record = weight + record_index * 640u;
    __global const float *scales = (__global const float *)record;
    __global const uchar *packed = record + 128u;
    const float2 scale = vload2(0, scales + column_lane);

    for (uint inner_lane = 0; inner_lane < 32u; ++inner_lane) {
      const uchar byte = packed[inner_lane * 16u + column_lane / 2u];
      const uint low = (uint)byte & 15u;
      const uint high = (uint)byte >> 4;
      const float2 quantized =
          (float2)(low < 8u ? (int)low : (int)low - 16,
                   high < 8u ? (int)high : (int)high - 16);
      const uint k = inner_tile * 32u + inner_lane;
      const float activation =
          convert_float(input[(size_t)row * inner + k]);
      accumulator = fma((float2)(activation) * quantized, scale, accumulator);
    }
  }
  vstore_half2_rte(accumulator, 0,
                   output + (size_t)row * columns + column);
}

// Four adjacent outputs per work-item. The two byte loads exactly cover four
// signed INT4 values and the activation is reused across four FP32 FMAs.
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void w4a16_linear_quad_n32k32(
    __global const half *restrict input,
    __global const uchar *restrict weight,
    __global half *restrict output,
    const uint rows,
    const uint inner,
    const uint columns) {
  const uint quad = get_global_id(0);
  const uint row = get_global_id(1);
  const uint column = quad * 4u;
  if (row >= rows || column >= columns)
    return;

  const uint column_tile = column / 32u;
  const uint column_lane = column & 31u;
  const uint inner_tiles = inner / 32u;
  float4 accumulator = (float4)(0.0f);
  for (uint inner_tile = 0; inner_tile < inner_tiles; ++inner_tile) {
    const size_t record_index =
        (size_t)column_tile * (size_t)inner_tiles + inner_tile;
    __global const uchar *record = weight + record_index * 640u;
    __global const float *scales = (__global const float *)record;
    __global const uchar *packed = record + 128u;
    const float4 scale = vload4(0, scales + column_lane);

    for (uint inner_lane = 0; inner_lane < 32u; ++inner_lane) {
      const uint byte_index = inner_lane * 16u + column_lane / 2u;
      const uchar low_byte = packed[byte_index];
      const uchar high_byte = packed[byte_index + 1u];
      const uint q0 = (uint)low_byte & 15u;
      const uint q1 = (uint)low_byte >> 4;
      const uint q2 = (uint)high_byte & 15u;
      const uint q3 = (uint)high_byte >> 4;
      const float4 quantized =
          (float4)(q0 < 8u ? (int)q0 : (int)q0 - 16,
                   q1 < 8u ? (int)q1 : (int)q1 - 16,
                   q2 < 8u ? (int)q2 : (int)q2 - 16,
                   q3 < 8u ? (int)q3 : (int)q3 - 16);
      const uint k = inner_tile * 32u + inner_lane;
      const float activation =
          convert_float(input[(size_t)row * inner + k]);
      accumulator = fma((float4)(activation) * quantized, scale, accumulator);
    }
  }
  vstore_half4_rte(accumulator, 0,
                   output + (size_t)row * columns + column);
}

// Eight outputs match Mali-G610's preferred FP16 vector width. Four adjacent
// bytes and one activation feed an eight-lane FP32 accumulator.
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void w4a16_linear_octet_n32k32(
    __global const half *restrict input,
    __global const uchar *restrict weight,
    __global half *restrict output,
    const uint rows,
    const uint inner,
    const uint columns) {
  const uint octet = get_global_id(0);
  const uint row = get_global_id(1);
  const uint column = octet * 8u;
  if (row >= rows || column >= columns)
    return;

  const uint column_tile = column / 32u;
  const uint column_lane = column & 31u;
  const uint inner_tiles = inner / 32u;
  float8 accumulator = (float8)(0.0f);
  for (uint inner_tile = 0; inner_tile < inner_tiles; ++inner_tile) {
    const size_t record_index =
        (size_t)column_tile * (size_t)inner_tiles + inner_tile;
    __global const uchar *record = weight + record_index * 640u;
    __global const float *scales = (__global const float *)record;
    __global const uchar *packed = record + 128u;
    const float8 scale = vload8(0, scales + column_lane);

    for (uint inner_lane = 0; inner_lane < 32u; ++inner_lane) {
      const uint byte_index = inner_lane * 16u + column_lane / 2u;
      const uchar4 bytes = vload4(0, packed + byte_index);
      const uint q0 = (uint)bytes.s0 & 15u;
      const uint q1 = (uint)bytes.s0 >> 4;
      const uint q2 = (uint)bytes.s1 & 15u;
      const uint q3 = (uint)bytes.s1 >> 4;
      const uint q4 = (uint)bytes.s2 & 15u;
      const uint q5 = (uint)bytes.s2 >> 4;
      const uint q6 = (uint)bytes.s3 & 15u;
      const uint q7 = (uint)bytes.s3 >> 4;
      const float8 quantized =
          (float8)(q0 < 8u ? (int)q0 : (int)q0 - 16,
                   q1 < 8u ? (int)q1 : (int)q1 - 16,
                   q2 < 8u ? (int)q2 : (int)q2 - 16,
                   q3 < 8u ? (int)q3 : (int)q3 - 16,
                   q4 < 8u ? (int)q4 : (int)q4 - 16,
                   q5 < 8u ? (int)q5 : (int)q5 - 16,
                   q6 < 8u ? (int)q6 : (int)q6 - 16,
                   q7 < 8u ? (int)q7 : (int)q7 - 16);
      const uint k = inner_tile * 32u + inner_lane;
      const float activation =
          convert_float(input[(size_t)row * inner + k]);
      accumulator = fma((float8)(activation) * quantized, scale, accumulator);
    }
  }
  vstore_half8_rte(accumulator, 0,
                   output + (size_t)row * columns + column);
}

// Encoder/batched path: one 32x8 workgroup computes an 8-row by 32-column
// output tile. Each K32 packed weight tile is dequantized once into local
// memory and reused across eight rows; each activation tile is loaded once
// instead of once per output column.
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void w4a16_linear_batched_n32k32(
    __global const half *restrict input,
    __global const uchar *restrict weight,
    __global half *restrict output,
    const uint rows,
    const uint inner,
    const uint columns) {
  const uint column_tile = get_group_id(0);
  const uint row_tile = get_group_id(1);
  const uint column_lane = get_local_id(0);
  const uint row_lane = get_local_id(1);
  const uint local_linear = row_lane * 32u + column_lane;
  const uint column = column_tile * 32u + column_lane;
  const uint row = row_tile * 8u + row_lane;
  const uint inner_tiles = inner / 32u;
  __local half activation_tile[8][32];
  __local float weight_tile[32][32];
  float accumulator = 0.0f;

  for (uint inner_tile = 0; inner_tile < inner_tiles; ++inner_tile) {
    const uint activation_row = local_linear / 32u;
    const uint activation_inner = local_linear & 31u;
    const uint global_row = row_tile * 8u + activation_row;
    activation_tile[activation_row][activation_inner] =
        global_row < rows
            ? input[(size_t)global_row * inner + inner_tile * 32u +
                    activation_inner]
            : convert_half(0.0f);

    const size_t record_index =
        (size_t)column_tile * (size_t)inner_tiles + inner_tile;
    __global const uchar *record = weight + record_index * 640u;
    __global const float *scales = (__global const float *)record;
    __global const uchar *packed = record + 128u;
#pragma unroll
    for (uint load = 0; load < 4u; ++load) {
      const uint packed_linear = local_linear + load * 256u;
      const uint k_lane = packed_linear / 32u;
      const uint n_lane = packed_linear & 31u;
      const uchar byte = packed[packed_linear >> 1];
      const uint nibble =
          (packed_linear & 1u) ? ((uint)byte >> 4) : ((uint)byte & 15u);
      const int quantized = nibble < 8u ? (int)nibble : (int)nibble - 16;
      weight_tile[k_lane][n_lane] =
          (float)quantized * scales[n_lane];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (uint k_lane = 0; k_lane < 32u; ++k_lane) {
      accumulator = fma(convert_float(activation_tile[row_lane][k_lane]),
                        weight_tile[k_lane][column_lane], accumulator);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (row < rows && column < columns)
    output[(size_t)row * columns + column] = convert_half_rte(accumulator);
}

// Barrier-free batched path. One work-item owns a 4x8 output tile, reusing
// every dequantized eight-column W4 vector across four activation rows.
__attribute__((reqd_work_group_size(16, 4, 1)))
__kernel void w4a16_linear_batched_octet4_n32k32(
    __global const half *restrict input,
    __global const uchar *restrict weight,
    __global half *restrict output,
    const uint rows,
    const uint inner,
    const uint columns) {
  const uint octet = get_global_id(0);
  const uint row_group = get_global_id(1);
  const uint column = octet * 8u;
  const uint row0 = row_group * 4u;
  const uint row1 = row0 + 1u;
  const uint row2 = row0 + 2u;
  const uint row3 = row0 + 3u;
  if (row0 >= rows || column >= columns)
    return;

  const uint column_tile = column / 32u;
  const uint column_lane = column & 31u;
  const uint inner_tiles = inner / 32u;
  float8 accumulator0 = (float8)(0.0f);
  float8 accumulator1 = (float8)(0.0f);
  float8 accumulator2 = (float8)(0.0f);
  float8 accumulator3 = (float8)(0.0f);
  for (uint inner_tile = 0; inner_tile < inner_tiles; ++inner_tile) {
    const size_t record_index =
        (size_t)column_tile * (size_t)inner_tiles + inner_tile;
    __global const uchar *record = weight + record_index * 640u;
    __global const float *scales = (__global const float *)record;
    __global const uchar *packed = record + 128u;
    const float8 scale = vload8(0, scales + column_lane);

    for (uint inner_lane = 0; inner_lane < 32u; ++inner_lane) {
      const uint byte_index = inner_lane * 16u + column_lane / 2u;
      const uchar4 bytes = vload4(0, packed + byte_index);
      const uint q0 = (uint)bytes.s0 & 15u;
      const uint q1 = (uint)bytes.s0 >> 4;
      const uint q2 = (uint)bytes.s1 & 15u;
      const uint q3 = (uint)bytes.s1 >> 4;
      const uint q4 = (uint)bytes.s2 & 15u;
      const uint q5 = (uint)bytes.s2 >> 4;
      const uint q6 = (uint)bytes.s3 & 15u;
      const uint q7 = (uint)bytes.s3 >> 4;
      const float8 quantized =
          (float8)(q0 < 8u ? (int)q0 : (int)q0 - 16,
                   q1 < 8u ? (int)q1 : (int)q1 - 16,
                   q2 < 8u ? (int)q2 : (int)q2 - 16,
                   q3 < 8u ? (int)q3 : (int)q3 - 16,
                   q4 < 8u ? (int)q4 : (int)q4 - 16,
                   q5 < 8u ? (int)q5 : (int)q5 - 16,
                   q6 < 8u ? (int)q6 : (int)q6 - 16,
                   q7 < 8u ? (int)q7 : (int)q7 - 16);
      const half8 dequantized = convert_half8_rte(quantized * scale);
      const uint k = inner_tile * 32u + inner_lane;
      const half activation0 = input[(size_t)row0 * inner + k];
      accumulator0 += convert_float8((half8)(activation0) * dequantized);
      if (row1 < rows) {
        const half activation1 = input[(size_t)row1 * inner + k];
        accumulator1 += convert_float8((half8)(activation1) * dequantized);
      }
      if (row2 < rows) {
        const half activation2 = input[(size_t)row2 * inner + k];
        accumulator2 += convert_float8((half8)(activation2) * dequantized);
      }
      if (row3 < rows) {
        const half activation3 = input[(size_t)row3 * inner + k];
        accumulator3 += convert_float8((half8)(activation3) * dequantized);
      }
    }
  }
  vstore_half8_rte(accumulator0, 0,
                   output + (size_t)row0 * columns + column);
  if (row1 < rows)
    vstore_half8_rte(accumulator1, 0,
                     output + (size_t)row1 * columns + column);
  if (row2 < rows)
    vstore_half8_rte(accumulator2, 0,
                     output + (size_t)row2 * columns + column);
  if (row3 < rows)
    vstore_half8_rte(accumulator3, 0,
                     output + (size_t)row3 * columns + column);
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

std::string device_string(cl_device_id device, cl_device_info key) {
  size_t bytes = 0;
  check(clGetDeviceInfo(device, key, 0, nullptr, &bytes),
        "clGetDeviceInfo(size)");
  std::vector<char> value(bytes);
  check(clGetDeviceInfo(device, key, value.size(), value.data(), nullptr),
        "clGetDeviceInfo(value)");
  if (!value.empty() && value.back() == '\0')
    value.pop_back();
  return std::string(value.begin(), value.end());
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

size_t checked_bytes(size_t first, size_t second, size_t element_size) {
  if (first == 0 || second == 0 ||
      first > std::numeric_limits<size_t>::max() / second ||
      first * second > std::numeric_limits<size_t>::max() / element_size) {
    throw std::runtime_error("OpenCL tensor allocation size overflow");
  }
  return first * second * element_size;
}

void run_w4(OpenClW4Context &context, cl_mem weight, cl_mem input_buffer,
            cl_mem output_buffer, size_t input_columns,
            size_t output_columns, size_t max_rows,
            const llmc_float16 *input, llmc_float16 *output, size_t rows,
            OpenClW4Metrics &metrics) {
  if (input == nullptr || output == nullptr || rows == 0 || rows > max_rows)
    throw std::runtime_error("invalid OpenCL W4 run arguments");
  if (rows > std::numeric_limits<cl_uint>::max() ||
      input_columns > std::numeric_limits<cl_uint>::max() ||
      output_columns > std::numeric_limits<cl_uint>::max()) {
    throw std::runtime_error("OpenCL W4 dimensions exceed uint32");
  }

  const size_t input_bytes =
      checked_bytes(rows, input_columns, sizeof(llmc_float16));
  const size_t output_bytes =
      checked_bytes(rows, output_columns, sizeof(llmc_float16));
  const auto wall_start = Clock::now();
  cl_event write_event = nullptr;
  cl_event kernel_event = nullptr;
  cl_event read_event = nullptr;
  check(clEnqueueWriteBuffer(context.queue(), input_buffer, CL_FALSE, 0,
                             input_bytes, input, 0, nullptr, &write_event),
        "clEnqueueWriteBuffer(input)");

  const bool split_inner =
      rows == 1 && (input_columns >= 4096 || output_columns <= 5120);
  cl_kernel kernel = rows != 1          ? context.batched_octet4_kernel()
                     : split_inner      ? context.split_kernel()
                                        : context.octet_kernel();
  const cl_uint row_count = static_cast<cl_uint>(rows);
  const cl_uint inner = static_cast<cl_uint>(input_columns);
  const cl_uint columns = static_cast<cl_uint>(output_columns);
  check(clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer),
        "clSetKernelArg(input)");
  check(clSetKernelArg(kernel, 1, sizeof(weight), &weight),
        "clSetKernelArg(weight)");
  check(clSetKernelArg(kernel, 2, sizeof(output_buffer), &output_buffer),
        "clSetKernelArg(output)");
  check(clSetKernelArg(kernel, 3, sizeof(row_count), &row_count),
        "clSetKernelArg(rows)");
  check(clSetKernelArg(kernel, 4, sizeof(inner), &inner),
        "clSetKernelArg(inner)");
  check(clSetKernelArg(kernel, 5, sizeof(columns), &columns),
        "clSetKernelArg(columns)");

  const size_t local[2] = {
      rows != 1 ? size_t{16} : split_inner ? kTileColumns : size_t{128},
      rows != 1 ? size_t{4} : split_inner ? kLocalInnerSplits : size_t{1}};
  const size_t octet_columns =
      (output_columns / 8 + size_t{127}) / size_t{128} * size_t{128};
  const size_t batched_octets =
      (output_columns / 8 + local[0] - 1) / local[0] * local[0];
  const size_t batched_row_groups =
      ((rows + 3) / 4 + local[1] - 1) / local[1] * local[1];
  const size_t global[2] = {
      rows != 1 ? batched_octets
                : split_inner ? output_columns : octet_columns,
      rows != 1 ? batched_row_groups
                : split_inner ? kLocalInnerSplits : rows};
  check(clEnqueueNDRangeKernel(context.queue(), kernel, 2, nullptr, global,
                               local, 1, &write_event, &kernel_event),
        "clEnqueueNDRangeKernel(W4)");
  check(clEnqueueReadBuffer(context.queue(), output_buffer, CL_FALSE, 0,
                            output_bytes, output, 1, &kernel_event,
                            &read_event),
        "clEnqueueReadBuffer(output)");
  check(clWaitForEvents(1, &read_event), "clWaitForEvents(read)");

  metrics.calls += 1;
  metrics.kernel_ms += event_ms(kernel_event);
  metrics.transfer_ms += event_ms(write_event) + event_ms(read_event);
  metrics.wall_ms += std::chrono::duration<double, std::milli>(
                         Clock::now() - wall_start)
                         .count();
  clReleaseEvent(read_event);
  clReleaseEvent(kernel_event);
  clReleaseEvent(write_event);
}

} // namespace

OpenClW4Context::OpenClW4Context() {
  cl_uint platform_count = 0;
  check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs");
  if (platform_count == 0)
    throw std::runtime_error("no OpenCL platform");
  std::vector<cl_platform_id> platforms(platform_count);
  check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
        "clGetPlatformIDs(list)");

  for (cl_platform_id platform : platforms) {
    cl_uint count = 0;
    cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_,
                                   &count);
    if (status == CL_SUCCESS && count != 0)
      break;
    device_ = nullptr;
  }
  if (device_ == nullptr)
    throw std::runtime_error("no OpenCL GPU device");

  device_name_ = device_string(device_, CL_DEVICE_NAME);
  check(clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(compute_units_), &compute_units_, nullptr),
        "clGetDeviceInfo(compute units)");
  const std::string extensions = device_string(device_, CL_DEVICE_EXTENSIONS);
  if (extensions.find("cl_khr_fp16") == std::string::npos)
    throw std::runtime_error("OpenCL GPU lacks cl_khr_fp16");

  cl_int status = CL_SUCCESS;
  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &status);
  check(status, "clCreateContext");
  queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE,
                                &status);
  check(status, "clCreateCommandQueue");
  const size_t source_bytes = std::char_traits<char>::length(kW4KernelSource);
  const char *source = kW4KernelSource;
  program_ = clCreateProgramWithSource(context_, 1, &source,
                                       &source_bytes, &status);
  check(status, "clCreateProgramWithSource");
  status = clBuildProgram(program_, 1, &device_, "-cl-fast-relaxed-math",
                          nullptr, nullptr);
  if (status != CL_SUCCESS) {
    size_t log_bytes = 0;
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_bytes);
    std::vector<char> log(log_bytes + 1, '\0');
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_bytes,
                          log.data(), nullptr);
    throw std::runtime_error("OpenCL W4 kernel build failed: " +
                             std::string(log.data()));
  }
  split_kernel_ =
      clCreateKernel(program_, "w4a16_linear_n32k32", &status);
  check(status, "clCreateKernel");
  pair_kernel_ =
      clCreateKernel(program_, "w4a16_linear_pair_n32k32", &status);
  check(status, "clCreateKernel(pair)");
  quad_kernel_ =
      clCreateKernel(program_, "w4a16_linear_quad_n32k32", &status);
  check(status, "clCreateKernel(quad)");
  octet_kernel_ =
      clCreateKernel(program_, "w4a16_linear_octet_n32k32", &status);
  check(status, "clCreateKernel(octet)");
  batched_kernel_ =
      clCreateKernel(program_, "w4a16_linear_batched_n32k32", &status);
  check(status, "clCreateKernel(batched)");
  batched_octet4_kernel_ = clCreateKernel(
      program_, "w4a16_linear_batched_octet4_n32k32", &status);
  check(status, "clCreateKernel(batched octet4)");
}

OpenClW4Context::~OpenClW4Context() {
  if (batched_octet4_kernel_ != nullptr)
    clReleaseKernel(batched_octet4_kernel_);
  if (batched_kernel_ != nullptr)
    clReleaseKernel(batched_kernel_);
  if (octet_kernel_ != nullptr)
    clReleaseKernel(octet_kernel_);
  if (quad_kernel_ != nullptr)
    clReleaseKernel(quad_kernel_);
  if (pair_kernel_ != nullptr)
    clReleaseKernel(pair_kernel_);
  if (split_kernel_ != nullptr)
    clReleaseKernel(split_kernel_);
  if (program_ != nullptr)
    clReleaseProgram(program_);
  if (queue_ != nullptr)
    clReleaseCommandQueue(queue_);
  if (context_ != nullptr)
    clReleaseContext(context_);
}

OpenClW4Linear::OpenClW4Linear(OpenClW4Context &context,
                               const W4Tensor &weight, size_t max_rows)
    : context_(context), input_columns_(weight.shape[1]),
      output_columns_(weight.shape[0]), max_rows_(max_rows) {
  if (weight.encoding != W4Encoding::kW4A16GroupTiled ||
      weight.dimensions != 2 || input_columns_ % kTileInner != 0 ||
      output_columns_ % kTileColumns != 0 || max_rows_ == 0) {
    throw std::runtime_error("OpenCL W4 linear requires tiled N32/K32 weight");
  }

  cl_int status = CL_SUCCESS;
  weight_ = clCreateBuffer(context_.context(),
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           weight.data_size, const_cast<void *>(weight.data),
                           &status);
  check(status, "clCreateBuffer(weight)");
  input_ = clCreateBuffer(
      context_.context(), CL_MEM_READ_ONLY,
      checked_bytes(max_rows_, input_columns_, sizeof(llmc_float16)), nullptr,
      &status);
  check(status, "clCreateBuffer(input)");
  output_ = clCreateBuffer(
      context_.context(), CL_MEM_WRITE_ONLY,
      checked_bytes(max_rows_, output_columns_, sizeof(llmc_float16)), nullptr,
      &status);
  check(status, "clCreateBuffer(output)");
}

OpenClW4Linear::~OpenClW4Linear() {
  if (output_ != nullptr)
    clReleaseMemObject(output_);
  if (input_ != nullptr)
    clReleaseMemObject(input_);
  if (weight_ != nullptr)
    clReleaseMemObject(weight_);
}

void OpenClW4Linear::run(const llmc_float16 *input, llmc_float16 *output,
                         size_t rows) {
  run_w4(context_, weight_, input_, output_, input_columns_, output_columns_,
         max_rows_, input, output, rows, metrics_);
}

OpenClW4LinearPlan::OpenClW4LinearPlan(OpenClW4Context &context, size_t rows,
                                       size_t input_columns,
                                       size_t output_columns)
    : context_(context), rows_(rows), input_columns_(input_columns),
      output_columns_(output_columns) {
  if (rows_ == 0 || input_columns_ == 0 || input_columns_ % kTileInner != 0 ||
      output_columns_ == 0 || output_columns_ % kTileColumns != 0) {
    throw std::runtime_error("invalid OpenCL W4 plan shape");
  }
  cl_int status = CL_SUCCESS;
  input_ = clCreateBuffer(
      context_.context(), CL_MEM_READ_ONLY,
      checked_bytes(rows_, input_columns_, sizeof(llmc_float16)), nullptr,
      &status);
  check(status, "clCreateBuffer(plan input)");
  output_ = clCreateBuffer(
      context_.context(), CL_MEM_WRITE_ONLY,
      checked_bytes(rows_, output_columns_, sizeof(llmc_float16)), nullptr,
      &status);
  check(status, "clCreateBuffer(plan output)");
}

OpenClW4LinearPlan::~OpenClW4LinearPlan() {
  if (output_ != nullptr)
    clReleaseMemObject(output_);
  if (input_ != nullptr)
    clReleaseMemObject(input_);
}

OpenClW4LinearPlan::Weight::~Weight() {
  if (buffer_ != nullptr)
    clReleaseMemObject(buffer_);
}

std::unique_ptr<OpenClW4LinearPlan::Weight>
OpenClW4LinearPlan::prepare(const W4Tensor &weight) {
  if (weight.encoding != W4Encoding::kW4A16GroupTiled ||
      weight.dimensions != 2 || weight.shape[1] != input_columns_ ||
      weight.shape[0] != output_columns_) {
    throw std::runtime_error("OpenCL W4 weight does not match plan");
  }
  std::unique_ptr<Weight> result(new Weight());
  cl_int status = CL_SUCCESS;
  result->buffer_ = clCreateBuffer(
      context_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      weight.data_size, const_cast<void *>(weight.data), &status);
  check(status, "clCreateBuffer(plan weight)");
  result->bytes_ = weight.data_size;
  return result;
}

void OpenClW4LinearPlan::run(const Weight &weight,
                             const llmc_float16 *input,
                             llmc_float16 *output) {
  if (weight.buffer_ == nullptr)
    throw std::runtime_error("invalid OpenCL W4 prepared weight");
  run_w4(context_, weight.buffer_, input_, output_, input_columns_,
         output_columns_, rows_, input, output, rows_, metrics_);
}

} // namespace llmc
