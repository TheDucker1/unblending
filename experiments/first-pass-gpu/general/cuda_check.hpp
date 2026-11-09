#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

inline void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
