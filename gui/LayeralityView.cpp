#include "LayeralityView.hpp"
#include <vector>

namespace Ui {
	LayeralityView::LayeralityView(int id) {
		auto treeNodeText = "Layer " + std::to_string(id);
		mTreeNode = TreeNode::Ptr(new TreeNode(treeNodeText, [=] {

			ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,1 });
			mRmLr->display();
			ImGui::SameLine();
			mAddSplv->display();
			ImGui::SameLine();
			mAddCell->display();
			ImGui::PopStyleVar();

			for (const auto& cellularity : m_cellularitys) {
				cellularity->view();
			}
			for (const auto& spilloverity : m_spilloveritys) {
				spilloverity->view();
			}
		}));

		mRmLr    = RmButton::Ptr(new RmButton("lr"));
		mRmLr->clicked.connect([&](){ m_isShouldBeRemoved = true; });

		mAddSplv = AddButton::Ptr(new AddButton("splv"));
		mAddCell = AddButton::Ptr(new AddButton("cell"));
	}

	void LayeralityView::view() const {
		mTreeNode->display();
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
	bool LayeralityView::isShouldBeRemoved() const {
		return m_isShouldBeRemoved;
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
			
			ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,1 });
			mAddLr->display();
			ImGui::SameLine();
			mSizePopupBtn->display();
			ImGui::PopStyleVar();

			for (auto& it: m_layeralitys) {
				it->view();
			}

			auto it = m_layeralitys.begin();
			while (it != m_layeralitys.end()) {
				it->get()->isShouldBeRemoved() ? it = m_layeralitys.erase(it) : ++it;
			}
		}));

		mAddLr = AddButton::Ptr(new AddButton("lr"));
		mAddLr->clicked.connect([&]() {
			LayeralityView::Ptr lw(new LayeralityView(m_layeralitys.size()));
			m_layeralitys.push_back(std::move(lw));
		});
	}

	void RegionView::view() const {
		mTreeNode->display();
	}

	void RegionView::load(instance::iregion& region) {
		m_size = region.innate().size;
		int id = 0;
		for (const auto& layer : region.innate().layers) {
			LayeralityView::Ptr lw(new LayeralityView(id++));
			lw->load(layer);

			m_layeralitys.push_back(std::move(lw));
		}

		mSizePopupBtn->setText(getSizeAsText());
	}

	std::string RegionView::getSizeAsText() const {
		return "width:" + std::to_string(m_size.width) + " " +
			   "height:" + std::to_string(m_size.height);
	}

}