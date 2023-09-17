#pragma once
#include "../core/terminality.hpp"

namespace Ui {
	class TerminalityView {
	public:
		using Ptr = std::unique_ptr<TerminalityView>;

		void view() const;
		void load(const instance::readable_trmn_innate& trmn);

	private:
		std::tuple<UPTR_TEMPLATE_TR> m_innate{nullptr, nullptr};
	};
}