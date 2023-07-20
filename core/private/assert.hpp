#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static void assert_err(cudaError_t code) {
	if (code != cudaError::cudaSuccess) {
		const char* file = __FILE__;
		int line = __LINE__;

		fprintf(stderr, "gpu_assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

static void assert_last_err() {
#ifdef _DEBUG
	assert_err(cudaPeekAtLastError());
#endif
}

