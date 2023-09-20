#include "controls.hpp"
#include "imgui_internal.h"

namespace Ui {
    std::atomic_int Control::mIdCounter = 0;

	Control::Control(const std::string& text)
        : mText(text), mId(mIdCounter++) {
    }

    Control::Control(const std::string& text, std::unique_ptr<std::function<void()>>& clicked) : Control(text) {
        mOnClicked = std::move(clicked);
    }

	void Control::display() {
        ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Border, mStyle.border);
        ImGui::PushStyleColor(ImGuiCol_Button, mStyle.color);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, mStyle.hovered);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, mStyle.active);

        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FrameRounding, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FramePadding, mFramePadding);

        ImGui::PushID(mId);
        bool clicked = ImGui::Button(mText.c_str());
        ImGui::PopID();

        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(3);
	}

}
