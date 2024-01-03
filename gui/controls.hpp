#pragma once
#include <memory>
#include <atomic>
#include <functional>
#include <boost/signals2.hpp>

#include "SubControls.hpp"

namespace Ui {
	class Control {
	public:
		using Ptr = std::unique_ptr<Control>;

		Control(const std::string& text);

		virtual void display() = 0;
		void setText(const std::string& text);
		int nextId();

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

	template<typename T> class InputedPopupBtn : public PopupBtn {
	public:
		using Ptr = std::unique_ptr<InputedPopupBtn>;
		using Signal = boost::signals2::signal<void()>;

		using SetterData = SetterData<T>;
		using SetterDataPtr = std::shared_ptr<SetterData>;

		InputedPopupBtn(const std::string& text, const std::string& popupText, const std::vector<SetterDataPtr>& settersData);
		void display() override;

		Signal valueSetterUpdated;

	private:
		std::vector<SetterDataPtr> mSettersData;
	};

	template<typename T>
	InputedPopupBtn<T>::InputedPopupBtn(const std::string& text, const std::string& popupText, const std::vector<SetterDataPtr>& settersData)
		: PopupBtn(text, popupText, [&]() { for (auto& sd : mSettersData) {if(ValueSetterDisplay(*sd))valueSetterUpdated();}}), mSettersData(settersData) {}

	template<typename T>
	void InputedPopupBtn<T>::display() {
		PopupBtn::display();
	}

	typedef InputedPopupBtn<int> IntInPpBtn;
#define IntInPpBtnBp(b,c) IntInPpBtn::SetterDataPtr(new IntInPpBtn::SetterData{ "",b,c })



	template<typename SizeType> class SizeTypeInputedPopupBtn {
	public:
		using Ptr = std::shared_ptr<SizeTypeInputedPopupBtn>;

		SizeTypeInputedPopupBtn(SizeType& size);
		std::string getSizeAsText() const;

		void view();
		static Ptr create(SizeType& sizeType);

	protected:
		SizeType& m_size;
		IntInPpBtn::Ptr mSizePopupBtn;
	};

	template<typename SizeType>
	SizeTypeInputedPopupBtn<SizeType>::SizeTypeInputedPopupBtn(SizeType& size): m_size(size) {
		mSizePopupBtn = IntInPpBtn::Ptr(new InputedPopupBtn<int>(getSizeAsText(), "Config", {
				IntInPpBtnBp("width", m_size.width),
				IntInPpBtnBp("height", m_size.height)
			})
		);

		mSizePopupBtn->valueSetterUpdated.connect([&]() {
			mSizePopupBtn->setText(getSizeAsText());
		});

		mSizePopupBtn->setText(getSizeAsText());
	}

	template<typename SizeType>
	std::string SizeTypeInputedPopupBtn<SizeType>::getSizeAsText() const {
		return "width:" + std::to_string(m_size.width) + " " +
			   "height:" + std::to_string(m_size.height);
	}

	template<typename SizeType>
	void SizeTypeInputedPopupBtn<SizeType>::view() {
		mSizePopupBtn->display();
	}

	template<typename SizeType>
	std::shared_ptr<SizeTypeInputedPopupBtn<SizeType>> SizeTypeInputedPopupBtn<SizeType>::create(SizeType& sizeType) {
		return Ptr(new SizeTypeInputedPopupBtn<SizeType>(sizeType));
	}


	class ButtonEx : public Control {
	public:
		using Ptr = std::unique_ptr<ButtonEx>;
		using Signal = boost::signals2::signal<void()>;

		ButtonEx(const ImColor& color, const std::string& text);
		void display() override;

		Signal clicked;

	protected:
		std::string mText;
	};

	class AddButton: public ButtonEx {
	public:
		AddButton(const std::string& text = "");
	};

	class RmButton : public ButtonEx {
	public:
		RmButton(const std::string& text = "");
	};


}
