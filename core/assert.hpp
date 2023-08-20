#pragma once
#include <string>
#include <cuda_runtime.h>
#include "boost.hpp"

#define assert_err(code) {                        \
	if (code != cudaError::cudaSuccess) {         \
		const char* file = __FILE__;              \
		int line = __LINE__;                      \
		fprintf(stderr, "gpu_assert: %s %s %d\n", \
			cudaGetErrorString(code), file, line);\
		exit(code);                               \
	}                                             \
}                                                 \

void assert_last_err();

void console(std::string str);
void console(const ptree& root);

#define STRINGIZING(x) #x
#define STR(x) STRINGIZING(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

#define logexit() { console(FILE_LINE); exit(1); }

