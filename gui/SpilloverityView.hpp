#pragma once
#include "../core/spilloverity.hpp"

namespace Ui {
	class SpilloverityView {
	public:
		using Ptr = std::unique_ptr<SpilloverityView>;

		void view() const;
		void load(const instance::readable_splvr_innate& splvr);

	private:
		std::unique_ptr<innate::spillover> m_innate {nullptr};
	};
}
