#include "CellularityView.hpp"

namespace Ui {
	void CellularityView::view() const {

		for (const auto& terminality : m_terminalitys) {
			terminality->view();
		}
	}

	void CellularityView::load(const instance::readable_cell_innate& cell) {
		cell_data_tuple::create_first(cell.cell->type, [&](auto ptr){
			m_innate = std::move(ptr);
			memcpy(m_innate.get(), cell.cell, sizeof(*ptr));
		});

		for (const auto& terminality : cell.terminalitys) {
			TerminalityView::Ptr trw(new TerminalityView());
			trw->load(terminality);

			m_terminalitys.push_back(std::move(trw));
		}
	}

}