#include "SpilloverityView.hpp"

namespace Ui {
	SpilloverityView::SpilloverityView(
		instance::ispilloverity_host_accessor& accessor, const std::string& name)
			: m_accessor(accessor), m_name(name)
	{
		mSizeTypeInputedPopupBtn = SizeTypeInputedPopupBtn<innate::size>
			::create(const_cast<innate::size&>(accessor.spilloverity().size()), true);

		mRmSplvr = RmButton::Ptr(new RmButton("splvr"));
		mRmSplvr->clicked.connect([&]() { m_isShouldBeRemoved = true; });
	}

	void SpilloverityView::view() const {
		ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,1 });
		{
			mRmSplvr->display();
			ImGui::SameLine();

			mSizeTypeInputedPopupBtn->view();
			ImGui::SameLine();
		}
		ImGui::PopStyleVar();

		auto splvr = m_accessor.spilloverity().innate().splvr;
		hana::for_each(*splvr, hana::fuse([&](auto member, auto value) {
			std::string name = hana::to<char const*>(member);

		}));

		ImGui::Spacing();
	}

	std::string SpilloverityView::name() const {
		return m_name;
	}

	bool SpilloverityView::isShouldBeRemoved() const {
		return m_isShouldBeRemoved;
	}

}