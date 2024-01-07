#include "CellularityView.hpp"

namespace Ui {
	CellularityView::CellularityView(instance::icellularity_host_accessor& accessor, const std::string& name)
		: m_accessor(accessor), m_name("cell_" + name)
	{
		mSizeTypeInputedPopupBtn = SizeTypeInputedPopupBtn<innate::size>
			::create(const_cast<innate::size&>(accessor.cellularity().size()), true);

		mRmCell = RmButton::Ptr(new RmButton("cell"));
		mRmCell->clicked.connect([&]() { m_isShouldBeRemoved = true; });

		auto cell = m_accessor.cellularity().innate().cell;

		std::string static_text = "name:" + m_name + " ";
		hana::for_each(*cell, hana::fuse([&](auto member, auto value) {
			std::string name = hana::to<char const*>(member);
			static_text += name + ":" + innate::to_string(value) + " ";
		}));

		mStaticTextButton = TextButton::Ptr(new TextButton(static_text, true));
		mDynamicTextButton = TextButton::Ptr(new TextButton("", false));

		mTrParamsInputedPopupBtn = TrParamsInputedPopupBtn::Ptr(new TrParamsInputedPopupBtn());
		mTrParamsInputedPopupBtn->apply.connect([&](instance::iterminality::InnateTerminalityParam param) {
			auto name = to_hex_str(m_tr++);
			auto& acc = m_accessor.add_trmn(name, param);

			TerminalityView::Ptr lw(new TerminalityView(acc, name));
			m_terminalitys.push_back(std::move(lw));
		});

		std::unordered_map<std::string, instance::iterminality_host_accessor&> trmns;
		accessor.get_trmns(trmns);

		for (const auto& trmn : trmns) {
			TerminalityView::Ptr trw(new TerminalityView(trmn.second, trmn.first));
			m_terminalitys.push_back(std::move(trw));
		}
	}

	void CellularityView::view() {
		ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,1 });
		{
			mRmCell->display();
			ImGui::SameLine();

			mTrParamsInputedPopupBtn->display();
			ImGui::SameLine();

			mSizeTypeInputedPopupBtn->display();
			ImGui::SameLine();

		}
		ImGui::PopStyleVar();
		float x = ImGui::GetCursorPosX();

		mStaticTextButton->display();

		std::string dynamic_text;
		auto cell = m_accessor.cellularity().innate().cell;
		cell_data_tuple::to_first(const_cast<innate::cell*>(cell), [&](auto* t0) {
			hana::for_each(*t0, hana::fuse([&](auto member, auto value) {
				std::string name = hana::to<char const*>(member);
				dynamic_text += name + ":" + std::to_string(value) + " ";
				}));
			});

		if (!dynamic_text.empty()) {
			ImGui::SetCursorPos({ x, ImGui::GetCursorPosY() });
			mDynamicTextButton->setText(dynamic_text);
			mDynamicTextButton->display();
		}

		ImGui::SetNextItemOpen(true);
		if (!m_terminalitys.empty() && ImGui::TreeNode(("terminalitys " + m_name).c_str())) {
			for (const auto& terminality : m_terminalitys) {
				terminality->view();
			}
			auto it = m_terminalitys.begin();
			while (it != m_terminalitys.end()) {
				if (it->get()->isShouldBeRemoved())
				{
					m_accessor.rm_trmn(it->get()->name());
					it = m_terminalitys.erase(it);
				}
				else ++it;
			}
			ImGui::TreePop();
		}
	}

	std::string CellularityView::name() const {
		return m_name;
	}

	bool CellularityView::isShouldBeRemoved() const {
		return m_isShouldBeRemoved;
	}
}