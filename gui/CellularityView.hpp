#pragma once
#include "../core/cellularity.hpp"

namespace Ui {
	class TerminalityView;

	class CellulatityView {
	public:
		void view() const;

	private:
		std::unique_ptr<innate::cell> m_innate { nullptr };
		std::vector<TerminalityView> m_terminalitys;
	};

}