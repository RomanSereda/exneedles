#pragma once
#include <cuda_runtime.h>

void assert_err(cudaError_t code);
void assert_last_err();

