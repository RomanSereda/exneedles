#pragma once
#include <memory>
#include <atomic>
#include <functional>

#include "SubControls.hpp"

namespace Ui {
	class Control {
	public:
		using Ptr = std::unique_ptr<Control>;

		Control(const std::string& text);

		virtual void display() = 0;
		void setText(const std::string& text);

	protected:
		int mId = -1;
		static std::atomic_int mIdCounter;

		std::string mText;
		ControlStyle mStyle;
	};

	class Button: public Control {
	public:
		using Ptr = std::unique_ptr<Button>;

		Button(const std::string& text);
		Button(const std::string& text, const std::function<void()>& clicked);

		void display() override;

	protected:
		std::unique_ptr<std::function<void()>> mOnClicked{ nullptr };
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

	class PopupBtn : public Button {
	public:
		using Ptr = std::unique_ptr<PopupBtn>;

		PopupBtn(const std::string& text, const std::string& popupText, const std::function<void()>& popupContent);
		void display() override;

	private:
		Popup::Ptr mPopup;
	};
}
