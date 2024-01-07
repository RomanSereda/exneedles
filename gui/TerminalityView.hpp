#pragma once
#include "../core/terminality.hpp"

namespace Ui {
	class TerminalityView {
	public:
		using Ptr = std::unique_ptr<TerminalityView>;
		TerminalityView(instance::iterminality_host_accessor& accessor, const std::string& name);
		void view() const;

		std::string name() const;
		bool isShouldBeRemoved() const;

	private:
		instance::iterminality_host_accessor& m_accessor;

		std::string m_name;
		bool m_isShouldBeRemoved = false;

	};
}