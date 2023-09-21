#pragma once
#include <vector>
#include <string>

#include "Style.hpp"

#define SETTER_BUFFER_SIZE 256

namespace Ui {
	void StyleBegin(const ControlStyle& style);
	void StyleEnd(const ControlStyle& style);

	template<typename T> struct SetterData {
		T& value;
		std::string text;
		char buffer[SETTER_BUFFER_SIZE] = {0};
	};
	static int InputTextCallback(ImGuiInputTextCallbackData* data) {
		return 0;
	}
	template<typename T> void ValueSetterDisplay(SetterData<T>& data) {
		if (ImGui::InputText(data.text.c_str(), data.buffer, SETTER_BUFFER_SIZE, 0, &InputTextCallback, &data)) {
u			int r = 0;
		}
	}
	void IntValueSetter(int& value, const std::string& text);

	bool ButtonDisplay(const char* text, const ControlStyle& style);
}
