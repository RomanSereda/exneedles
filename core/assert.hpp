#pragma once
#include <string>
#include <cuda_runtime.h>

void assert_err(cudaError_t code);
void assert_last_err();

void console(std::string str);

#define STRINGIZING(x) #x
#define STR(x) STRINGIZING(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

#define logexit() { console(FILE_LINE); exit(1); }

