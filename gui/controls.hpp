#pragma once
#include <memory>
#include <atomic>
#include <functional>
#include <boost/signals2.hpp>

#include "SubControls.hpp"

#include "../core/spilloverity.hpp"
#include "../core/cellularity.hpp"
#include "../core/terminality.hpp"

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

		virtual void display() override;

	protected:
		std::unique_ptr<std::function<void()>> mOnClicked{ nullptr };
	};

	class Static : public Control {
	public:
		using Ptr = std::shared_ptr<Static>;

		Static();
		void display() override;
	};

	class CollapsingHeader {
	public:
		using Ptr = std::unique_ptr<CollapsingHeader>;

		CollapsingHeader(const std::string& text, const std::function<void()>& contentDisplay);
		bool display(bool collapsible = true);

	private:
		std::string mText;
		ControlStyle mStyle;
		std::function<void()> mContentDisplay;
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

		void open();
		bool running() const;

	private:
		bool mRunning = false;
		std::string mText;
		std::function<void()> mContentDisplay;

		ImVec4 mBorderColor = ImColor(0.6f, 0.6f, 0.9f, 1.0f);
	};

	class PopupBtn : public Button {
	public:
		using Ptr = std::unique_ptr<PopupBtn>;

		PopupBtn(const std::string& text, const std::string& popupText, const std::function<void()>& popupContent);
		void display() override;
		void setAdditionalContent(const std::function<void()>& additionalContent);

	private:
		Popup::Ptr mPopup;
		std::function<void()> mAdditionalContent;
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



	template<typename SizeType> class SizeTypeInputedPopupBtn: Control {
	public:
		using Ptr = std::shared_ptr<SizeTypeInputedPopupBtn>;

		SizeTypeInputedPopupBtn(SizeType& size, bool disabled = false);
		std::string getSizeAsText() const;

		void display() override;
		static Ptr create(SizeType& sizeType, bool disabled = false);

	protected:
		bool m_disabled;
		SizeType& m_size;

		IntInPpBtn::Ptr mSizePopupBtn;
	};

	template<typename SizeType>
	SizeTypeInputedPopupBtn<SizeType>::SizeTypeInputedPopupBtn(SizeType& size, bool disabled)
		: Control(""), m_size(size), m_disabled(disabled)
	{
		mSizePopupBtn = IntInPpBtn::Ptr(new InputedPopupBtn<int>(getSizeAsText(), "config_" + std::to_string(mId), {
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
	void SizeTypeInputedPopupBtn<SizeType>::display() {
		if (m_disabled) ImGui::BeginDisabled();

		mSizePopupBtn->display();

		if (m_disabled) ImGui::EndDisabled();
	}

	template<typename SizeType>
	std::shared_ptr<SizeTypeInputedPopupBtn<SizeType>> SizeTypeInputedPopupBtn<SizeType>::create(SizeType& sizeType, bool disabled) {
		return Ptr(new SizeTypeInputedPopupBtn<SizeType>(sizeType, disabled));
	}


	class ButtonEx : public Control {
	public:
		using Ptr = std::unique_ptr<ButtonEx>;
		using Signal = boost::signals2::signal<void()>;

		ButtonEx(const ImColor& color, const std::string& text);
		virtual void display() override;

		Signal clicked;
	};

	class AddButton: public ButtonEx {
	public:
		AddButton(const std::string& text = "");
	};

	class RmButton : public ButtonEx {
	public:
		RmButton(const std::string& text = "");
	};

	class TextButton : public ButtonEx {
	public:
		using Ptr = std::unique_ptr<TextButton>;

		TextButton(const std::string& text, bool disabled);

		void display() override;

	private:
		bool m_disabled = false;
	};


	template<typename T> class TypeSelectPopup: public Control {
	public:
		using Ptr = std::shared_ptr<TypeSelectPopup>;
		using Signal = boost::signals2::signal<void(T)>;

		TypeSelectPopup();

		void open(const std::string& text);
		void display() override;

		static Ptr create();
		bool running() const;

		Signal selected;

	protected:
		std::string mText;
		Popup::Ptr mPopup;
		int mSelectedIndex = 0;
	};

	template<typename T>
	TypeSelectPopup<T>::TypeSelectPopup(): Control(std::string()) {
		mPopup = Popup::Ptr(new Popup("Select" + std::to_string(mId), [&]() {
			std::vector<T> items;
			innate::get_items(items);

			int selectedIndex = 0;
			bool isSelected = false;

			if (ImGui::BeginCombo(mText.c_str(), ""/*innate::to_string(items[mSelectedIndex]).c_str() */ )) {
				for (int i = 0; i < items.size(); ++i) {
					if (ImGui::Selectable(innate::to_string(items[i]).c_str(), isSelected)) {
						selectedIndex = i;
						isSelected = true;
					}

					if (isSelected) {
						ImGui::SetItemDefaultFocus();
						mSelectedIndex = selectedIndex;

						break;
					}
				}
				ImGui::EndCombo();
			}

			if (isSelected) {
				selected(static_cast<T>(selectedIndex));
				ImGui::CloseCurrentPopup();
			}

		}));
	}

	template<typename T>
	void TypeSelectPopup<T>::open(const std::string& text) {
		mText = text;
		mPopup->open();
	}

	template<typename T>
	void TypeSelectPopup<T>::display(){
		mPopup->display();
	}

	template<typename T>
	std::shared_ptr<TypeSelectPopup<T>> TypeSelectPopup<T>::create() {
		return TypeSelectPopup<T>::Ptr(new TypeSelectPopup<T>());
	}

	template<typename T>
	bool TypeSelectPopup<T>::running() const {
		return mPopup->running();
	}


	class TrParamsInputedPopupBtn: Control
	{
	public:
		using Ptr = std::shared_ptr<TrParamsInputedPopupBtn>;
		using Signal = boost::signals2::signal<void(instance::iterminality::InnateTerminalityParam params)>;

		TrParamsInputedPopupBtn();
		void display() override;

		Signal apply;

	private:
		instance::iterminality::InnateTerminalityParam mParams;
		IntInPpBtn::Ptr mPopupBtn;

		int mTrSelectedIndex = 0;
		int mClSelectedIndex = 0;
	};

}
