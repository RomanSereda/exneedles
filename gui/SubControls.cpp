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
        SetterData<int> data = { value, text, "" };
        ValueSetterDisplay<int>(data);
    }

    bool ButtonDisplay(const char* text, const ControlStyle& style) {
        StyleBegin(style);
        bool result = ImGui::Button(text);
        StyleEnd(style);

        return result;
    }
}


