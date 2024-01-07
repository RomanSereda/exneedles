#include "TerminalityView.hpp"

namespace Ui {
	TerminalityView::TerminalityView(instance::iterminality_host_accessor& accessor, const std::string& name)
		: m_accessor(accessor), m_name(name)
	{}

	void TerminalityView::view() const {

	}

	std::string TerminalityView::name() const {
		return m_name;
	}

	bool TerminalityView::isShouldBeRemoved() const {
		return m_isShouldBeRemoved;
	}

	/*void TerminalityView::load(const instance::readable_trmn_innate& trmn) {
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
	}*/

}