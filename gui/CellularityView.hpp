#pragma once
#include "../core/cellularity.hpp"
#include "../core/layerality.hpp"
#include "Controls.hpp"

#include "TerminalityView.hpp"

namespace Ui {
	class TerminalityView;

	class CellularityView {
	public:
		using Ptr = std::unique_ptr<CellularityView>;
		
		CellularityView(instance::icellularity_host_accessor& accessor, const std::string& name);
		void view() const;

		std::string name() const;
		bool isShouldBeRemoved() const;

	private:
		instance::icellularity_host_accessor& m_accessor;
		std::string m_name;

		RmButton::Ptr  mRmCell;
		bool m_isShouldBeRemoved = false;

		SizeTypeInputedPopupBtn<innate::size>::Ptr mSizeTypeInputedPopupBtn;

		std::vector<TerminalityView::Ptr> m_terminalitys;
	};

}