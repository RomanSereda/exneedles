#include "LayeralityView.hpp"
#include <vector>

namespace Ui {
	LayeralityView::LayeralityView(instance::ilayerality_host_accessor& accessor, const std::string& name)
		: m_name(name)
	{
		mSizeTypeInputedPopupBtn = SizeTypeInputedPopupBtn<innate::size>
			::create(const_cast<innate::size&>(accessor.layerality().size()), true);

		auto treeNodeText = "lr: " + name;
		mTreeNode = TreeNode::Ptr(new TreeNode(treeNodeText, [=] {
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

		std::unordered_map<std::string, instance::icellularity_host_accessor&> cells;
		accessor.get_cells(cells);

		for (const auto& cell : cells) {
			CellularityView::Ptr sw(new CellularityView(cell.second, cell.first));
			m_cellularitys.push_back(std::move(sw));
		}

		std::unordered_map<std::string, instance::ispilloverity_host_accessor&> splvrs;
		accessor.get_splvrs(splvrs);

		for (const auto& splvr: splvrs) {
			SpilloverityView::Ptr splw(new SpilloverityView(splvr.second, splvr.first));
			m_spilloveritys.push_back(std::move(splw));
		}
	}

	void LayeralityView::view() const {
		ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,1 });
		{
			mRmLr->display();
			ImGui::SameLine();

			mAddSplv->display();
			ImGui::SameLine();

			mAddCell->display();
			ImGui::SameLine();

			mSizeTypeInputedPopupBtn->view();
			ImGui::SameLine();
		}
		ImGui::PopStyleVar();

		mTreeNode->display();
	}

	const std::string& LayeralityView::name() const {
		return m_name;
	}

	bool LayeralityView::isShouldBeRemoved() const {
		return m_isShouldBeRemoved;
	}
}

namespace Ui {
	RegionView::RegionView(instance::iregion_host_accessor& accessor, int id) {
		mSizeTypeInputedPopupBtn = SizeTypeInputedPopupBtn<innate::size>
			::create(const_cast<innate::size&>(accessor.region().size()));

		mAddLr = AddButton::Ptr(new AddButton("lr"));

		auto treeNodeText = id == -1 ? "rg" : "rg " + std::to_string(id);
		mTreeNode = TreeNode::Ptr(new TreeNode(treeNodeText, [&]{
			for (auto& it: m_layeralitys) {
				it->view();
			}

			auto it = m_layeralitys.begin();
			while (it != m_layeralitys.end()) {
				if (it->get()->isShouldBeRemoved()) 
				{
					accessor.rm_layer(it->get()->name());
					it = m_layeralitys.erase(it);
				} 
				else ++it;
			}
		}));
	
		mAddLr->clicked.connect([&]() {
			auto name = to_hex_str(m_lrid++);
			auto& acc = accessor.add_layer(name);

			LayeralityView::Ptr lw(new LayeralityView(acc, name));
			m_layeralitys.push_back(std::move(lw));
		});

		std::unordered_map<std::string, instance::ilayerality_host_accessor&> layers;
		accessor.get_layers(layers);

		for (const auto& layer : layers) {
			LayeralityView::Ptr lw(new LayeralityView(layer.second, layer.first));
			m_layeralitys.push_back(std::move(lw));
		}
	}

	void RegionView::view() const {
		ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ItemSpacing, { 2,1 });
		{
			mAddLr->display();
			ImGui::SameLine();

			mSizeTypeInputedPopupBtn->view();
			ImGui::SameLine();
		}
		ImGui::PopStyleVar();

		mTreeNode->display();
	}
}