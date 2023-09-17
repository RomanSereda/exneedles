#include "TerminalityView.hpp"

namespace Ui {
	void TerminalityView::view() const {
	}

	void TerminalityView::load(const instance::readable_trmn_innate& trmn) {
		std::unique_ptr<innate::cluster> cl;
		cluster_tuple::create(trmn.cl->type, [&](auto ptr) {
			cl = std::move(ptr);
			memcpy(cl.get(), trmn.cl, sizeof(*ptr));
		});

		std::unique_ptr<innate::terminal> tr;
		cluster_data_tuple::create_first(trmn.tr->type, [&](auto ptr) {
			tr = std::move(ptr);
			memcpy(tr.get(), trmn.tr, sizeof(*ptr));
		});
	}

}