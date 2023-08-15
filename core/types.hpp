#pragma once
#include <stdint.h>
#include <vector>
#include <memory>
#include <iostream>

typedef int8_t state8_t;
typedef int16_t state16_t;
typedef int32_t state32_t;

typedef uint8_t rgstr8_t;
typedef uint16_t rgstr16_t;
typedef uint32_t rgstr32_t;

#ifdef __CUDACC__
#define ALIGN(x)  __align__(x)
#else
#if defined(_MSC_VER) && (_MSC_VER >= 1300)
#define ALIGN(x) __declspec(align(x))
#else
#if defined(__GNUC__)
#define ALIGN(x)  __attribute__ ((aligned (x)))
#else
#define ALIGN(x)
#endif
#endif
#endif

#define __align_4b__ ALIGN(32)
#define __align_2b__ ALIGN(16)
#define __align_1b__ ALIGN(8)

#define __mem__
#define __const__

static void console(std::string str) {
	std::cout << str << std::endl;
}

#define STRINGIZING(x) #x
#define STR(x) STRINGIZING(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

static void logexit() {
	console(FILE_LINE);
	exit(1);
}
