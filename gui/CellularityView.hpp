#pragma once
#include "../core/cellularity.hpp"

namespace Ui {
	class TerminalityView;

	class CellulatityView {
	public:

	private:
		std::unique_ptr<innate::cell> m_innate;
		std::vector<TerminalityView> m_terminalitys;
	};

}