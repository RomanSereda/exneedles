#pragma once
#include "../core/spilloverity.hpp"

namespace Ui {
	class SpilloverityView {
	public:
		void view() const;

	private:
		std::unique_ptr<innate::spillover> m_innate {nullptr};
	};
}
