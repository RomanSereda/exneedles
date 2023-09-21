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

		void view() const;
		void load(const instance::readable_layer_innate& layer);

	private:
		std::vector<SpilloverityView::Ptr> m_spilloveritys;
		std::vector<CellularityView::Ptr>  m_cellularitys;
	};

	class RegionView {
	public:
		using Ptr = std::unique_ptr<RegionView>;

		RegionView(int id = -1);

		void view() const;
		void load(const instance::readable_region_innate& region);

	private:
		innate::size m_size;
		std::vector<LayeralityView::Ptr> m_layeralitys;

	private:
		TreeNode::Ptr mTreeNode;
		PopupBtn::Ptr mSizePopupBtn;

		std::string getSizeAsText() const;
	};

}