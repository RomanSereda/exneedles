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

		LayeralityView(int id);

		void view() const;
		void load(const instance::readable_layer_innate& layer);

		bool isShouldBeRemoved() const;

	private:
		std::vector<SpilloverityView::Ptr> m_spilloveritys;
		std::vector<CellularityView::Ptr>  m_cellularitys;

	private:
		TreeNode::Ptr mTreeNode;

		AddButton::Ptr mAddSplv;
		AddButton::Ptr mAddCell;
		RmButton::Ptr  mRmLr;

		bool m_isShouldBeRemoved = false;

	};

	class RegionView {
	public:
		using Ptr = std::unique_ptr<RegionView>;

		RegionView(int id = -1);

		void view() const;
		void load(instance::iregion& region);

	private:
		innate::size m_size;
		std::vector<LayeralityView::Ptr> m_layeralitys;

	private:
		TreeNode::Ptr mTreeNode;

		IntInPpBtn::Ptr mSizePopupBtn;
		AddButton::Ptr mAddLr;

		std::string getSizeAsText() const;
	};

}