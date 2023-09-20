#include "LayeralityView.hpp"

namespace Ui {
	void LayeralityView::view() const {
		for (const auto& cellularity : m_cellularitys) {
			cellularity->view();
		}
		for (const auto& spilloverity : m_spilloveritys) {
			spilloverity->view();
		}
	}

	void LayeralityView::load(const instance::readable_layer_innate& layer) {
		for (const auto& splvr : layer.spillovers) {
			SpilloverityView::Ptr splw(new SpilloverityView());
			splw->load(splvr);

			m_spilloveritys.push_back(std::move(splw));
		}

		for (const auto& cell : layer.cellularity) {
			CellularityView::Ptr sw(new CellularityView());
			sw->load(cell);

			m_cellularitys.push_back(std::move(sw));
		}
	}
}

namespace Ui {
	RegionView::RegionView(int id) {
		mRegionHeader = id == -1 ? std::make_unique<CollapsingHeader>("Region") : 
			                       std::make_unique<CollapsingHeader>("Region " + std::to_string(id));
	}

	void RegionView::view() const {
		mRegionHeader->display();

		for (const auto& layerality : m_layeralitys) {
			layerality->view();
		}
	}

	void RegionView::load(const instance::readable_region_innate& region) {
		m_size = region.size;
		for (const auto& layer : region.layers) {
			LayeralityView::Ptr lw(new LayeralityView());
			lw->load(layer);

			m_layeralitys.push_back(std::move(lw));
		}
	}

}