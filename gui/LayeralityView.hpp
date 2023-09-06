#pragma once
#include "../core/layerality.hpp"

namespace Ui {
	class CellularityView;
	class SpilloverityView;

	class LayeralityView {
	public:

	private:
		std::vector<SpilloverityView> m_spilloveritys;
		std::vector<CellularityView> m_cellularitys;
	};

	class RegionView {
	public:

	private:
		std::vector<LayeralityView> m_layeralitys;
	};

}