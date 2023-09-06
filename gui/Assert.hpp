#pragma once
#include "../core/boost.hpp"
#include <string>


void console(const std::string& str);
void console(const ptree& root);

#define STRINGIZING(x) #x
#define STR(x) STRINGIZING(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

#define logexit() { console(FILE_LINE); exit(1); }
