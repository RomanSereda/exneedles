#include "assert.hpp"

void console(const std::string& str) {
	std::cout << str << std::endl;
}

void console(const ptree& root) {
	console(boost::to_string(root));
}
