#include "SpilloverityView.hpp"

namespace Ui {
	SpilloverityView::SpilloverityView(
		instance::ispilloverity_host_accessor& accessor, const std::string& name)
			: m_accessor(accessor), m_name("splvr_" + name)
	{
		mSizeTypeInputedPopupBtn = SizeTypeInputedPopupBtn<innate::size>
			::create(const_cast<innate::size&>(accessor.spilloverity().size()), true);

		mRmSplvr = RmButton::Ptr(new RmButton("splvr"));
		mRmSplvr->clicked.connect([&]() { m_isShouldBeRemoved = true; });

		std::string text = "name:" + m_name + " ";
		auto splvr = m_accessor.spilloverity().innate().splvr;
		hana::for_each(*splvr, hana::fuse([&](auto member, auto value) {
			std::string name = hana::to<char const*>(member);
			text += name + ":" + innate::to_string(value);
		}));

		mTextButton = TextButton::Ptr(new TextButton(text, true));
	}

	void SpilloverityView::view() const {
		ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,1 });
		{
			mRmSplvr->display();
			ImGui::SameLine();

			mSizeTypeInputedPopupBtn->display();
			ImGui::SameLine();
		}
		ImGui::PopStyleVar();

		mTextButton->display();
	}

	std::string SpilloverityView::name() const {
		return m_name;
	}

	bool SpilloverityView::isShouldBeRemoved() const {
		return m_isShouldBeRemoved;
	}

}