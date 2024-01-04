#include "SpilloverityView.hpp"

namespace Ui {
	SpilloverityView::SpilloverityView(
		instance::ispilloverity_host_accessor& accessor, const std::string& name)
			: m_accessor(accessor), m_name(name)
	{}

	void SpilloverityView::view() const {
		auto splvr = m_accessor.spilloverity().innate().splvr;
		hana::for_each(*splvr, hana::fuse([&](auto member, auto value) {
			std::string name = hana::to<char const*>(member);

		}));
	}

}