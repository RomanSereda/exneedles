#pragma once
#include "../core/spilloverity.hpp"

namespace Ui {
	class SpilloverityView {
	public:
		using Ptr = std::unique_ptr<SpilloverityView>;

		SpilloverityView(instance::ispilloverity_host_accessor& accessor, const std::string& name);
		void view() const;

	private:
		instance::ispilloverity_host_accessor& m_accessor;
		std::string m_name;
	};
}
