#pragma once
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <atomic>

#include "imgui.h"

namespace Ui {
	struct ControlStyle {
		ImVec4 color = ImColor(0.2f, 0.2f, 0.2f, 1.0f);
		ImVec4 border = ImColor(0.5f, 0.5f, 0.5f, 1.0f);
		ImVec4 hovered = ImColor(0.3f, 0.3f, 0.3f, 1.0f);
		ImVec4 active = ImColor(0.4f, 0.4f, 0.4f, 1.0f);
	};

	class Control {
	public:
		using Ptr = std::unique_ptr<Control>;

		Control(const std::string& text);
		Control(const std::string& text, std::unique_ptr<std::function<void()>>& clicked);
		void display();

	private:
		int mId = -1;
		static std::atomic_int mIdCounter;

		std::string mText;
		std::unique_ptr<std::function<void()>> mOnClicked {nullptr};

		ControlStyle mStyle;

		ImVec2 mFramePadding = { 3.0f, 1.0f };
	};

	class CollapsingHeader {
	public:
		using Ptr = std::unique_ptr<CollapsingHeader>;

		CollapsingHeader(const std::string& text);
		bool display();

	private:
		std::string mText;
		ControlStyle mStyle;
	};

}
