#pragma once
#include "../core/layerality.hpp"
#include "../core/spilloverity.hpp"
#include "Controls.hpp"

namespace Ui {
	class SpilloverityView {
	public:
		using Ptr = std::unique_ptr<SpilloverityView>;

		SpilloverityView(instance::ispilloverity_host_accessor& accessor, const std::string& name);
		void view() const;

		std::string name() const;
		bool isShouldBeRemoved() const;

	private:
		instance::ispilloverity_host_accessor& m_accessor;
		std::string m_name;

	private:
		RmButton::Ptr  mRmSplvr;
		bool m_isShouldBeRemoved = false;
		
		TextButton::Ptr mTextButton;
		SizeTypeInputedPopupBtn<innate::size>::Ptr mSizeTypeInputedPopupBtn;
	};
}
