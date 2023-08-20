#include "assert.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "boost.hpp"

void assert_last_err() {
#ifdef _DEBUG
	console(cudaGetErrorString(cudaPeekAtLastError()));
#endif
}

void console(std::string str) {
	std::cout << str << std::endl;
}

void console(const ptree& root) {
	console(boost::to_string(root));
}
