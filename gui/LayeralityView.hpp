#pragma once
#include "../core/layerality.hpp"

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
		void view() const;
		void load(const instance::readable_region_innate& region);

	private:
		innate::size m_size;
		std::vector<LayeralityView::Ptr> m_layeralitys;
	};

}