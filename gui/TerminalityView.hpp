#pragma once
#include "../core/terminality.hpp"

namespace Ui {
	class TerminalityView {
	public:
		void view() const;

	private:
		std::tuple<UPTR_TEMPLATE_TR> m_innate{nullptr, nullptr};
	};
}