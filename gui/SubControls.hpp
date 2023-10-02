#pragma once
#include <vector>
#include <string>
#include <memory>
#include <type_traits>

#include "Style.hpp"

#define SETTER_BUFFER_SIZE 256

namespace Ui {
	static bool IsNumber(const std::string& s) {
		return(strspn(s.c_str(), "-.0123456789") == s.size());
	}

	void StyleBegin(const ControlStyle& style);
	void StyleEnd(const ControlStyle& style);

	template<typename T> struct SetterData {
		using Ptr = std::shared_ptr<SetterData<T>>;

		char buffer[SETTER_BUFFER_SIZE] = { 0 };
		std::string text;
		T& value;
	};

	template<typename T> static typename std::enable_if<std::is_same<T, int>::value, int>::type
		to_value(const char* text) { return atoi(text); }
	template<typename T> static typename std::enable_if<std::is_same<T, float>::value, float>::type
		to_value(const char* text) { return atof(text); }
	template<typename T> static typename std::enable_if<std::is_same<T, std::string>::value, std::string>::type
		to_value(const char* text) { return std::string(text); }

	enum StylePopperType {
		Color,
		Style
	};
	template<StylePopperType type, typename T1, typename T2> class StylePopper {
	public:
		StylePopper(const std::vector<std::pair<T1, T2>>& params, bool condition = true);
		~StylePopper();

	private:
		int mParamsSize = -1;
	};

	typedef StylePopper<StylePopperType::Color, ImGuiCol_, ImVec4> StyleColorPopper;

	static int InputTextCallback(ImGuiInputTextCallbackData* data) {
		return 0;
	}
	template<typename T> bool ValueSetterDisplay(SetterData<T>& data);

	void IntValueSetter(int& value, const std::string& text);

	bool ButtonDisplay(const char* text, const ControlStyle& style);
}
