cmake_minimum_required(VERSION 3.16)
project(whisper_rknpu2_llmc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH "$ORIGIN/lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

set(RKNN_TOOLKIT_ROOT "" CACHE PATH "Path to rknn-toolkit2 checkout")
set(RKNN_MODEL_ZOO_ROOT "" CACHE PATH "Path to rknn_model_zoo checkout")
set(RKLLM_SDK_ROOT "" CACHE PATH "Path to rknn-llm checkout")
if(NOT RKNN_TOOLKIT_ROOT OR NOT RKNN_MODEL_ZOO_ROOT)
  message(FATAL_ERROR "Set RKNN_TOOLKIT_ROOT and RKNN_MODEL_ZOO_ROOT")
endif()

set(RKNN_INCLUDE
    "${RKNN_TOOLKIT_ROOT}/rknpu2/runtime/Linux/librknn_api/include")
set(RKNN_LIBRARY
    "${RKNN_TOOLKIT_ROOT}/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so")
if(NOT EXISTS "${RKNN_LIBRARY}")
  # The benchmark deployment keeps the model-matched runtime beside the app.
  set(RKNN_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/lib/librknnrt.so")
endif()

find_library(FFTW3F_LIBRARY NAMES fftw3f REQUIRED)
find_library(SNDFILE_LIBRARY NAMES sndfile libsndfile.so.1 REQUIRED)
set(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
find_library(FFTW3F_STATIC_LIBRARY NAMES fftw3f REQUIRED)
set(CMAKE_FIND_LIBRARY_SUFFIXES "${SAVED_CMAKE_FIND_LIBRARY_SUFFIXES}")

add_executable(whisper_rknpu2_llmc
  whisper_rknpu2_llmc.cc
  "${RKNN_MODEL_ZOO_ROOT}/utils/audio_utils.c"
)

add_executable(w4a16_package_test
  w4a16_package_test.cc
  w4a16_model.cc
)
target_compile_options(w4a16_package_test PRIVATE -O3 -DNDEBUG)

add_executable(pcm16_wav_test
  pcm16_wav_test.cc
  pcm16_wav.cc
)
target_compile_options(pcm16_wav_test PRIVATE -O3 -DNDEBUG)

add_executable(rkllm_no_rope_patch
  rkllm_no_rope_patch.cc
)
target_compile_options(rkllm_no_rope_patch PRIVATE -O2 -DNDEBUG)

add_executable(w4a16_matmul_smoke
  w4a16_matmul_smoke.cc
  static_npu_scheduler.cc
  w4a16_matmul.cc
  w4a16_model.cc
)
target_include_directories(w4a16_matmul_smoke PRIVATE "${RKNN_INCLUDE}")
target_compile_options(w4a16_matmul_smoke PRIVATE -O3 -DNDEBUG)
target_link_libraries(w4a16_matmul_smoke PRIVATE
  "${RKNN_LIBRARY}"
  pthread
  dl
)
set_target_properties(w4a16_matmul_smoke PROPERTIES
  BUILD_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH "$ORIGIN/../lib"
)

find_package(OpenCL QUIET)
if(OpenCL_FOUND)
  add_executable(opencl_w4a16_smoke
    opencl_w4a16_smoke.cc
    opencl_w4a16.cc
    w4a16_model.cc
  )
  target_compile_options(opencl_w4a16_smoke PRIVATE -O3 -DNDEBUG)
  target_link_libraries(opencl_w4a16_smoke PRIVATE OpenCL::OpenCL pthread)

  add_executable(whisper_opencl_w4a16
    whisper_rknpu2_w4a16.cc
    opencl_attention.cc
    opencl_w4a16.cc
    static_npu_scheduler.cc
    w4a16_matmul.cc
    w4a16_model.cc
    pcm16_wav.cc
  )
  target_compile_definitions(whisper_opencl_w4a16 PRIVATE
    LLMC_OPENCL_ENABLED=1
    LLMC_DEFAULT_HYBRID_LINEAR=1
    LLMC_DEFAULT_HYBRID_ATTENTION=1
  )
  target_include_directories(whisper_opencl_w4a16 PRIVATE
    "${RKNN_INCLUDE}"
    "${RKNN_MODEL_ZOO_ROOT}/3rdparty/fftw/include"
  )
  target_compile_options(whisper_opencl_w4a16 PRIVATE -O3 -DNDEBUG)
  target_link_libraries(whisper_opencl_w4a16 PRIVATE
    OpenCL::OpenCL
    "${RKNN_LIBRARY}"
    "${FFTW3F_STATIC_LIBRARY}"
    m
    pthread
    dl
  )
  set_target_properties(whisper_opencl_w4a16 PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
  )
endif()

add_executable(whisper_rknpu2_w4a16
  whisper_rknpu2_w4a16.cc
  static_npu_scheduler.cc
  w4a16_matmul.cc
  w4a16_model.cc
  pcm16_wav.cc
)
target_include_directories(whisper_rknpu2_w4a16 PRIVATE
  "${RKNN_INCLUDE}"
  "${RKNN_MODEL_ZOO_ROOT}/3rdparty/fftw/include"
)
target_compile_options(whisper_rknpu2_w4a16 PRIVATE -O3 -DNDEBUG)
target_link_libraries(whisper_rknpu2_w4a16 PRIVATE
  "${RKNN_LIBRARY}"
  "${FFTW3F_STATIC_LIBRARY}"
  m
  pthread
  dl
)
set_target_properties(whisper_rknpu2_w4a16 PROPERTIES
  BUILD_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH "$ORIGIN/../lib"
)
target_include_directories(whisper_rknpu2_llmc PRIVATE
  "${RKNN_INCLUDE}"
  "${RKNN_MODEL_ZOO_ROOT}/utils"
  "${RKNN_MODEL_ZOO_ROOT}/3rdparty/libsndfile/include"
)
target_compile_options(whisper_rknpu2_llmc PRIVATE -O3 -DNDEBUG)
target_link_libraries(whisper_rknpu2_llmc PRIVATE
  "${RKNN_LIBRARY}"
  "${FFTW3F_LIBRARY}"
  "${SNDFILE_LIBRARY}"
  m
  pthread
  dl
)

if(RKLLM_SDK_ROOT)
  set(RKLLM_INCLUDE
      "${RKLLM_SDK_ROOT}/rkllm-runtime/Linux/librkllm_api/include")
  set(RKLLM_LIBRARY
      "${RKLLM_SDK_ROOT}/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so")
  if(NOT EXISTS "${RKLLM_LIBRARY}")
    message(FATAL_ERROR "Missing RKLLM runtime: ${RKLLM_LIBRARY}")
  endif()

  function(add_whisper_rkllm_runner target optimized)
    add_executable(${target}
      whisper_rkllm_w8a8.cc
      "${RKNN_MODEL_ZOO_ROOT}/utils/audio_utils.c"
    )
    target_compile_definitions(${target} PRIVATE
      LLMC_RKLLM_OPTIMIZED=${optimized}
    )
    target_include_directories(${target} PRIVATE
      "${RKNN_INCLUDE}"
      "${RKLLM_INCLUDE}"
      "${RKNN_MODEL_ZOO_ROOT}/utils"
      "${RKNN_MODEL_ZOO_ROOT}/3rdparty/libsndfile/include"
    )
    target_compile_options(${target} PRIVATE -O3 -DNDEBUG)
    target_link_libraries(${target} PRIVATE
      "${RKLLM_LIBRARY}"
      "${RKNN_LIBRARY}"
      "${FFTW3F_LIBRARY}"
      "${SNDFILE_LIBRARY}"
      m
      pthread
      dl
    )
    set_target_properties(${target} PROPERTIES
      BUILD_RPATH "$ORIGIN/lib"
      INSTALL_RPATH "$ORIGIN/lib"
    )
  endfunction()

  add_whisper_rkllm_runner(whisper_rkllm_baseline 0)
  add_whisper_rkllm_runner(whisper_rkllm_llmc 1)
endif()
