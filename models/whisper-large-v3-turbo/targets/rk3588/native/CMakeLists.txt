cmake_minimum_required(VERSION 3.16)
project(whisper_rknpu2_llmc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH "$ORIGIN/lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

set(RKNN_TOOLKIT_ROOT "" CACHE PATH "Path to rknn-toolkit2 checkout")
set(RKNN_MODEL_ZOO_ROOT "" CACHE PATH "Path to rknn_model_zoo checkout")
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

add_executable(whisper_rknpu2_llmc
  whisper_rknpu2_llmc.cc
  "${RKNN_MODEL_ZOO_ROOT}/utils/audio_utils.c"
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
