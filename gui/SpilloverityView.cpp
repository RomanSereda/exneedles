#include "SpilloverityView.hpp"

namespace Ui {
	void SpilloverityView::view() const {
	}

	void SpilloverityView::load(const instance::readable_splvr_innate& splvr) {
		spillover_data_tuple::create_first(splvr.splvr->type, [&](auto ptr) {
			m_innate = std::move(ptr);
			memcpy(m_innate.get(), splvr.splvr, sizeof(*ptr));
		});
	}

}