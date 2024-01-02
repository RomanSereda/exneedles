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
		mSizePopupBtn = IntInPpBtn::Ptr(new InputedPopupBtn<int>(getSizeAsText(), "Config", {
				IntInPpBtnBp("width", m_size.width),
				IntInPpBtnBp("height", m_size.height)
			})
		);

		mSizePopupBtn->valueSetterUpdated.connect([&]() {
			mSizePopupBtn->setText(getSizeAsText());
		});

		auto treeNodeText = id == -1 ? "Region" : "Region " + std::to_string(id);
		mTreeNode = TreeNode::Ptr(new TreeNode(treeNodeText, [=]{
			mSizePopupBtn->display();
		}));

		mAddRmButton = AddRmButton::Ptr(new AddRmButton(true));
		mAddRmButton->addClicked.connect([&]() {
			LayeralityView::Ptr lw(new LayeralityView());
			m_layeralitys.push_back(std::move(lw));
		});
	}

	void RegionView::view() const {
		mTreeNode->display();
		mAddRmButton->display();

		for (const auto& layerality : m_layeralitys) {
			layerality->view();
		}
	}

	void RegionView::load(instance::iregion& region) {
		m_size = region.innate().size;
		for (const auto& layer : region.innate().layers) {
			LayeralityView::Ptr lw(new LayeralityView());
			lw->load(layer);

			m_layeralitys.push_back(std::move(lw));
		}

		mSizePopupBtn->setText(getSizeAsText());
	}

	std::string RegionView::getSizeAsText() const
	{
		return "width:" + std::to_string(m_size.width) + " " +
			   "height:" + std::to_string(m_size.height);
	}

}