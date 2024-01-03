#pragma once
#include <memory>
#include "../core/layerality.hpp"

#include "Controls.hpp"
#include "CellularityView.hpp"
#include "SpilloverityView.hpp"

namespace Ui {
	class LayeralityView {
	public:
		using Ptr = std::unique_ptr<LayeralityView>;

		LayeralityView(const std::string& name);

		void view() const;
		void load(const instance::readable_layer_innate& layer);

		const std::string& name() const;
		bool isShouldBeRemoved() const;

	private:
		std::vector<SpilloverityView::Ptr> m_spilloveritys;
		std::vector<CellularityView::Ptr>  m_cellularitys;

	private:
		TreeNode::Ptr mTreeNode;

		RmButton::Ptr  mRmLr;
		AddButton::Ptr mAddSplv;
		AddButton::Ptr mAddCell;

		std::string mName;
		bool m_isShouldBeRemoved = false;

	};

	class RegionView {
	public:
		using Ptr = std::unique_ptr<RegionView>;

		RegionView(instance::iregion& region, int id = -1);
		void view() const;

	private:
		int m_lrid = 0;
		std::vector<LayeralityView::Ptr> m_layeralitys;

	private:
		TreeNode::Ptr mTreeNode;
		AddButton::Ptr mAddLr;

		SizeTypeInputedPopupBtn<innate::size>::Ptr mSizeTypeInputedPopupBtn;
	};

}