#include "SubControls.hpp"

namespace Ui {
    void StyleBegin(const ControlStyle& style) {
        ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Border, style.border);
        ImGui::PushStyleColor(ImGuiCol_Button, style.color);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, style.hovered);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.active);

        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FrameRounding, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FramePadding, style.framePadding);
    }

    void StyleEnd(const ControlStyle& style) {
        ImGui::PopStyleColor(style.StyleColorCount);
        ImGui::PopStyleVar(style.StyleVarCount);
    }

    void IntValueSetter(int& value, const std::string& text) {
        SetterData<int> data = { "", text, value};

        auto strValue = std::to_string(value);
        memcpy(data.buffer, strValue.c_str(), strValue.length());

        ValueSetterDisplay(data);
    }

    bool ButtonDisplay(const char* text, const ControlStyle& style) {
        StyleBegin(style);
        bool result = ImGui::Button(text);
        StyleEnd(style);

        return result;
    }

	template<StylePopperType type, typename T1, typename T2>
	StylePopper<type, T1, T2>::StylePopper(const std::vector<std::pair<T1, T2>>& params, bool condition) {
		if (!condition)
			return;

		mParamsSize = params.size();
		for (const auto& param : params) {
			switch (type) {
			case Ui::Color:
				ImGui::PushStyleColor(param.first, param.second);
				break;
			case Ui::Style:
				break;
			}
		}
	}

	template<StylePopperType type, typename T1, typename T2>
	StylePopper<type, T1, T2>::~StylePopper() {
		if (mParamsSize == -1)
			return;

		switch (type) {
		case Ui::Color:
			ImGui::PopStyleColor(mParamsSize);
			break;
		case Ui::Style:
			break;
		}
	}

	template<typename T>
	bool ValueSetterDisplay(SetterData<T>& data) {
		StyleColorPopper inputStyle({ std::make_pair(ImGuiCol_FrameBg, EditedInputColor) },
			data.buffer != std::to_string(data.value));

		if (ImGui::InputText(data.text.c_str(), data.buffer, SETTER_BUFFER_SIZE, ImGuiInputTextFlags_EnterReturnsTrue, &InputTextCallback, &data)) {
			bool pressedEnter = ImGui::IsItemFocused() && ImGui::IsKeyPressed(ImGuiKey_Enter);
			if (pressedEnter) {
				data.value = to_value<T>(data.buffer);
				return true;
			}
		}
		return false;
	}
}


