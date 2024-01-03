#include "LayeralityView.hpp"
#include <vector>

namespace Ui {
	LayeralityView::LayeralityView(const std::string& name)
		: mName(name)
	{
		auto treeNodeText = "lr: " + name;
		mTreeNode = TreeNode::Ptr(new TreeNode(treeNodeText, [=] {

			ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,1 });
			{
				mRmLr->display();
				ImGui::SameLine();
				mAddSplv->display();
				ImGui::SameLine();
				mAddCell->display();
			}
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
	const std::string& LayeralityView::name() const {
		return mName;
	}

	bool LayeralityView::isShouldBeRemoved() const {
		return m_isShouldBeRemoved;
	}
}

namespace Ui {
	RegionView::RegionView(instance::iregion& region, int id) {
		mSizeTypeInputedPopupBtn = SizeTypeInputedPopupBtn<innate::size>::create(const_cast<innate::size&>(region.size()));

		auto treeNodeText = id == -1 ? "Region" : "Region " + std::to_string(id);
		mTreeNode = TreeNode::Ptr(new TreeNode(treeNodeText, [=]{
			
			ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,1 });
			{
				mAddLr->display();

				ImGui::SameLine();
				mSizeTypeInputedPopupBtn->view();
			}
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
			LayeralityView::Ptr lw(new LayeralityView(to_hex_str(m_lrid++)));
			m_layeralitys.push_back(std::move(lw));
		});

		for (const auto& layer : region.innate().layers) {
			LayeralityView::Ptr lw(new LayeralityView(to_hex_str(m_lrid++)));
			lw->load(layer);

			m_layeralitys.push_back(std::move(lw));
		}
	}

	void RegionView::view() const {
		mTreeNode->display();
	}
}