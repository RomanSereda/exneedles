#include "SpilloverityView.hpp"

namespace Ui {
	void SpilloverityView::view() const {
		hana::for_each(*m_innate.get(), hana::fuse([&](auto member, auto value) {
			std::string name = hana::to<char const*>(member);

		}));
	}

	void SpilloverityView::load(const instance::readable_splvr_innate& splvr) {
		spillover_data_tuple::create_first(splvr.splvr->type, [&](auto ptr) {
			m_innate = std::move(ptr);
			memcpy(m_innate.get(), splvr.splvr, sizeof(*ptr));
		});
	}

}