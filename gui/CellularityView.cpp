#include "CellularityView.hpp"

namespace Ui {
	CellularityView::CellularityView(instance::icellularity_host_accessor& accessor, const std::string& name)
		: m_accessor(accessor), m_name(name)
	{
		mSizeTypeInputedPopupBtn = SizeTypeInputedPopupBtn<innate::size>
			::create(const_cast<innate::size&>(accessor.cellularity().size()), true);

		mRmCell = RmButton::Ptr(new RmButton("cell"));
		mRmCell->clicked.connect([&]() { m_isShouldBeRemoved = true; });



		std::unordered_map<std::string, instance::iterminality_host_accessor&> trmns;
		accessor.get_trmns(trmns);

		for (const auto& trmn : trmns) {
			TerminalityView::Ptr trw(new TerminalityView(trmn.second, trmn.first));
			m_terminalitys.push_back(std::move(trw));
		}
	}

	void CellularityView::view() const {
		ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,1 });
		{
			mRmCell->display();
			ImGui::SameLine();

			mSizeTypeInputedPopupBtn->view();
			ImGui::SameLine();
		}
		ImGui::PopStyleVar();


		for (const auto& terminality : m_terminalitys) {
			terminality->view();
		}

		ImGui::Spacing();
	}

	std::string CellularityView::name() const {
		return m_name;
	}

	bool CellularityView::isShouldBeRemoved() const {
		return m_isShouldBeRemoved;
	}

	/*void CellularityView::load(const instance::readable_cell_innate& cell) {
		cell_data_tuple::create_first(cell.cell->type, [&](auto ptr){
			m_innate = std::move(ptr);
			memcpy(m_innate.get(), cell.cell, sizeof(*ptr));
		});

		for (const auto& terminality : cell.terminalitys) {
			TerminalityView::Ptr trw(new TerminalityView());
			trw->load(terminality);

			m_terminalitys.push_back(std::move(trw));
		}
	}*/

}