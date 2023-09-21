#pragma once
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <atomic>

#include "imgui.h"

namespace Ui {
	const ImVec4 ClearColor = ImColor(0.1f, 0.1f, 0.1f, 1.0f);

	struct ControlStyle {
		ImVec4 color = ImColor(0.2f, 0.2f, 0.2f, 1.0f);
		ImVec4 border = ImColor(0.5f, 0.5f, 0.5f, 1.0f);
		ImVec4 hovered = ImColor(0.3f, 0.3f, 0.3f, 1.0f);
		ImVec4 active = ImColor(0.4f, 0.4f, 0.4f, 1.0f);

		ImVec2 framePadding = { 3.0f, 2.0f };
	};

#define SETTER_BUFFER_SIZE 256
	template<typename T> struct SetterData {
		T& value;
		std::string text;
		char buffer[SETTER_BUFFER_SIZE];
	};
	static int InputTextCallback(ImGuiInputTextCallbackData* data) {
		return 0;
	}
	template<typename T> void ValueSetter(SetterData<T>& data) {
		ImGui::InputText(data.text.c_str(), data.buffer, SETTER_BUFFER_SIZE, 0, &InputTextCallback, &data);
	}

	class Control {
	public:
		using Ptr = std::unique_ptr<Control>;
		Control();
		Control(const std::string& text);
		Control(const std::function<void()>& clicked);
		Control(const std::string& text, const std::function<void()>& clicked);

		virtual void display();
		static bool display(const char* text, const ControlStyle& style);

		void setText(const std::string& text);

	protected:
		std::unique_ptr<std::function<void()>> mOnClicked{ nullptr };

	private:
		int mId = -1;
		static std::atomic_int mIdCounter;

		std::string mText;
		ControlStyle mStyle;
	};

	class CollapsingHeader {
	public:
		using Ptr = std::unique_ptr<CollapsingHeader>;

		CollapsingHeader(const std::string& text);
		bool display(bool collapsible = true);

	private:
		std::string mText;
		ControlStyle mStyle;
	};

	class TreeNode {
	public:
		using Ptr = std::unique_ptr<TreeNode>;

		TreeNode(const std::string& text, const std::function<void()>& contentDisplay);
		void display();

	private:
		std::string mText;
		std::function<void()> mContentDisplay;
	};

	class Popup {
	public:
		using Ptr = std::unique_ptr<Popup>;

		Popup(const std::string& text, const std::function<void()>& contentDisplay);
		void display();

		void open() const;

	private:
		std::string mText;
		std::function<void()> mContentDisplay;

		ImVec4 mBorderColor = ImColor(0.6f, 0.6f, 0.9f, 1.0f);
	};

	class PopupBtn : public Control {
	public:
		using Ptr = std::unique_ptr<PopupBtn>;

		PopupBtn(const std::string& text, const std::string& popupText, const std::function<void()>& popupContent);
		void display() override;

	private:
		Popup::Ptr mPopup;
	};
}
