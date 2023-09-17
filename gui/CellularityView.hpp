#pragma once
#include "../core/cellularity.hpp"

#include "TerminalityView.hpp"

namespace Ui {
	class TerminalityView;

	class CellularityView {
	public:
		using Ptr = std::unique_ptr<CellularityView>;
		
		void view() const;
		void load(const instance::readable_cell_innate& cell);

	private:
		std::unique_ptr<innate::cell> m_innate { nullptr };
		std::vector<TerminalityView::Ptr> m_terminalitys;
	};

}