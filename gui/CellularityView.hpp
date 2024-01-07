#pragma once
#include "../core/cellularity.hpp"
#include "../core/layerality.hpp"
#include "Controls.hpp"

#include "TerminalityView.hpp"

namespace Ui {
	class TerminalityView;

	class CellularityView {
	public:
		using Ptr = std::unique_ptr<CellularityView>;
		
		CellularityView(instance::icellularity_host_accessor& accessor, const std::string& name);
		void view();

		std::string name() const;
		bool isShouldBeRemoved() const;

	private:
		int m_tr = 0;
		instance::icellularity_host_accessor& m_accessor;

		std::string m_name;
		bool m_isShouldBeRemoved = false;

		TreeNode::Ptr mTreeNode;
		RmButton::Ptr mRmCell;

		TextButton::Ptr mStaticTextButton;
		TextButton::Ptr mDynamicTextButton;

		TrParamsInputedPopupBtn::Ptr mTrParamsInputedPopupBtn;
		SizeTypeInputedPopupBtn<innate::size>::Ptr mSizeTypeInputedPopupBtn;

		std::vector<TerminalityView::Ptr> m_terminalitys;

	};

}