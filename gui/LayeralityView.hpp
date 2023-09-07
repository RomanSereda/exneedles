#pragma once
#include "../core/layerality.hpp"

namespace Ui {
	class CellularityView;
	class SpilloverityView;

	class LayeralityView {
	public:
		void view() const;

	private:
		std::vector<SpilloverityView> m_spilloveritys;
		std::vector<CellularityView>  m_cellularitys;
	};

	class RegionView {
	public:
		void view() const;

	private:
		innate::size m_size;
		std::vector<LayeralityView> m_layeralitys;
	};

}