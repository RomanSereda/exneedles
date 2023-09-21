#include "controls.hpp"
#include "imgui_internal.h"

namespace Ui {
    std::atomic_int Control::mIdCounter = 0;

    Control::Control(): Control(""){}
    Control::Control(const std::function<void()>& clicked) : Control("", clicked) {}

    Control::Control(const std::string& text)
        : mText(text), mId(mIdCounter++) {
        mStyle.active = ImColor(1.0f, 0.6f, 0.0f, 1.0f);
    }

    Control::Control(const std::string& text, const std::function<void()>& clicked) : Control(text) {
        mOnClicked = std::make_unique<std::function<void()>>(clicked);
    }

	void Control::display() {
        ImGui::PushID(mId);
        bool clicked = display(mText.c_str(), mStyle);
        ImGui::PopID();

        if (clicked && mOnClicked) (*mOnClicked)();
	}

    bool Control::display(const char* text, const ControlStyle& style)
    {
        ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Border, style.border);
        ImGui::PushStyleColor(ImGuiCol_Button, style.color);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, style.hovered);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.active);

        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FrameRounding, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FramePadding, style.framePadding);

        bool result = ImGui::Button(text);

        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(3);

        return result;
    }

    void Control::setText(const std::string& text) {
        mText = text;
    }

    CollapsingHeader::CollapsingHeader(const std::string& text): mText(text) {
    }

    bool CollapsingHeader::display(bool collapsible)
    {
        ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Header, mStyle.color);
        ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Border, mStyle.border);
        ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_HeaderHovered, mStyle.hovered);
        ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_HeaderActive, mStyle.active);

        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FrameRounding, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FramePadding, { 1.0f, 1.0f });
        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,3 });

        ImGui::SetNextItemOpen(true, ImGuiCond_::ImGuiCond_FirstUseEver);

        auto flag = collapsible ? ImGuiTreeNodeFlags_AllowItemOverlap : 
                                  ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_Leaf;

        bool collapsed = ImGui::CollapsingHeader(mText.c_str(), flag);

        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(4);

        return collapsed;
    }


    TreeNode::TreeNode(const std::string& text, const std::function<void()>& contentDisplay)
        : mText(text), mContentDisplay(contentDisplay) {
    }
    
    void TreeNode::display() {
        if (ImGui::TreeNode(mText.c_str())) {
            mContentDisplay();
            ImGui::TreePop();
        }
    }


    Popup::Popup(const std::string& text, const std::function<void()>& contentDisplay) 
        : mText(text), mContentDisplay(contentDisplay) {
    }

    void Popup::display() {
        
        ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Border, mBorderColor);
        ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_PopupBg, ClearColor);
        ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_PopupRounding, 1.0);

        if(ImGui::BeginPopup(mText.c_str())) {
            mContentDisplay();
            ImGui::EndPopup();
        }

        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(1);
    }

    void Popup::open() const {
        ImGui::OpenPopup(mText.c_str());
    }

    PopupBtn::PopupBtn(const std::string& text, const std::string& popupText, const std::function<void()>& popupContent)
        : Control(text) {
        mPopup = Popup::Ptr(new Popup(popupText, [popupContent]{
            popupContent();
        }));

        mOnClicked = std::make_unique<std::function<void()>>([&]{
            mPopup->open();
        });
    }

    void PopupBtn::display() {
        Control::display();
        mPopup->display();
    }

}
