#pragma once
#include "../core/cellularity.hpp"

#include "TerminalityView.hpp"

namespace Ui {
	class TerminalityView;

	class CellularityView {
	public:
		using Ptr = std::unique_ptr<CellularityView>;
		
		CellularityView(instance::icellularity_host_accessor& accessor, const std::string& name);
		void view() const;


	private:
	private:
		instance::icellularity_host_accessor& m_accessor;
		std::string m_name;

		std::vector<TerminalityView::Ptr> m_terminalitys;
	};

}