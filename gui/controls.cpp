#include "controls.hpp"
#include "imgui_internal.h"

namespace Ui {
    std::atomic_int Control::mIdCounter = 0;

    Control::Control(const std::string& text)
        : mText(text), mId(mIdCounter++) {
        mStyle.active = ImColor(1.0f, 0.6f, 0.0f, 1.0f);
    }

    void Control::setText(const std::string& text) {
        mText = text;
    }

    int Control::nextId() {
        return ++mIdCounter;
    }

    Button::Button(const std::string& text) : Control(text){}

    Button::Button(const std::string& text, const std::function<void()>& clicked): Button(text){
        mOnClicked = std::make_unique<std::function<void()>>(clicked);
    }

    void Button::display() {
        ImGui::PushID(mId);
        bool clicked = ButtonDisplay(mText.c_str(), mStyle);
        ImGui::PopID();

        if (clicked && mOnClicked) (*mOnClicked)();
    }


    CollapsingHeader::CollapsingHeader(const std::string& text, const std::function<void()>& contentDisplay)
        : mText(text), mContentDisplay(contentDisplay) {}

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

        mContentDisplay();

        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(4);

        return collapsed;
    }


    TreeNode::TreeNode(const std::string& text, const std::function<void()>& contentDisplay)
        : mText(text), mContentDisplay(contentDisplay) {
    }
    
    void TreeNode::display() {
        ImGui::SetNextItemOpen(true);
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

        if (ImGui::BeginPopup(mText.c_str())) {
            mContentDisplay();
            ImGui::EndPopup();
        }
        else mRunning = false;

        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(1);
    }

    void Popup::open() {
        mRunning = true;
        ImGui::OpenPopup(mText.c_str());
    }

    bool Popup::running() const {
        return mRunning;
    }

    PopupBtn::PopupBtn(const std::string& text, const std::string& popupText, 
                       const std::function<void()>& popupContent) : Button(text) {
        mPopup = Popup::Ptr(new Popup(popupText, [popupContent]{
            popupContent();
        }));

        mOnClicked = std::make_unique<std::function<void()>>([&]{
            mPopup->open();
        });
    }

    void PopupBtn::display() {
        Button::display();
        mPopup->display();
    }


    ButtonEx::ButtonEx(const ImColor& color, const std::string& text)
        : Control(std::string()), mText(text) 
    {
        mStyle.color = color;
    }

    void ButtonEx::display() {
        ImGui::PushID(mId);
        bool clickedAdd = ButtonDisplay(mText.c_str(), mStyle);
        ImGui::PopID();

        if (clickedAdd) clicked();
    }

    AddButton::AddButton(const std::string& text)
        : ButtonEx(ImColor(0.2f, 0.25f, 0.2f, 1.0f), text.empty() ? "+add" : "+add " + text)
    {}

    RmButton::RmButton(const std::string& text)
        : ButtonEx(ImColor(0.25f, 0.2f, 0.2f, 1.0f), text.empty() ? "-rm" : "-rm " + text)
    {}

}
