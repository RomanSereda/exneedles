#include "LayeralityView.hpp"
#include <vector>

namespace Ui {
	LayeralityView::LayeralityView(instance::ilayerality_host_accessor& accessor, const std::string& name)
		: m_name("lr_" + name)
	{
		mSizeTypeInputedPopupBtn = SizeTypeInputedPopupBtn<innate::size>
			::create(const_cast<innate::size&>(accessor.layerality().size()), true);

		mCellTypeSelectPopup = TypeSelectPopup<innate::cell::cell_type>::create();
		mSplvrTypeSelectPopup = TypeSelectPopup<innate::spillover::spillover_type>::create();

		mTreeNode = TreeNode::Ptr(new TreeNode(m_name, [&] {
			cellularityView(accessor);
			spilloverityView(accessor);
		}));

		mRmLr    = RmButton::Ptr(new RmButton("lr"));
		mRmLr->clicked.connect([&](){ m_isShouldBeRemoved = true; });

		addCellInit(accessor);
		addSplvInit(accessor);

		std::unordered_map<std::string, instance::icellularity_host_accessor&> cells;
		accessor.get_cells(cells);

		for (const auto& cell : cells) {
			CellularityView::Ptr lw(new CellularityView(cell.second, cell.first));
			m_cellularitys.push_back(std::move(lw));
		}

		cellularityLoad(accessor);
		spilloverityLoad(accessor);
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

			mSizeTypeInputedPopupBtn->display();
			ImGui::SameLine();
		}
		ImGui::PopStyleVar();

		mTreeNode->display();

		mCellTypeSelectPopup->display();
		mSplvrTypeSelectPopup->display();
	}

	const std::string& LayeralityView::name() const {
		return m_name;
	}

	bool LayeralityView::isShouldBeRemoved() const {
		return m_isShouldBeRemoved;
	}

	void LayeralityView::addCellInit(instance::ilayerality_host_accessor& accessor) {
		mAddCell = AddButton::Ptr(new AddButton("cell"));
		mAddCell->clicked.connect([&]() {
			if (mCellTypeSelectPopup->running()) return;

			auto name = to_hex_str(m_cellId++);

			mCellTypeSelectPopup->open("cell: " + name);
			mCellTypeSelectPopup->selected.connect([&, name](auto type) {
				mCellTypeSelectPopup->selected.disconnect_all_slots();

				auto& acc = accessor.add_cell(name, type);

				CellularityView::Ptr lw(new CellularityView(acc, name));
				m_cellularitys.push_back(std::move(lw));;
			});
		});
	}

	void LayeralityView::addSplvInit(instance::ilayerality_host_accessor& accessor) {

		mAddSplv = AddButton::Ptr(new AddButton("splv"));
		mAddSplv->clicked.connect([&]() {
			if (mSplvrTypeSelectPopup->running()) return;

			auto name = to_hex_str(m_splvrId++);

			mSplvrTypeSelectPopup->open("splvr: " + name);
			mSplvrTypeSelectPopup->selected.connect([&, name](auto type) {
				mSplvrTypeSelectPopup->selected.disconnect_all_slots();

				auto& acc = accessor.add_splvr(name, type);

				SpilloverityView::Ptr lw(new SpilloverityView(acc, name));
				m_spilloveritys.push_back(std::move(lw));
				});
			});

		std::unordered_map<std::string, instance::ispilloverity_host_accessor&> splvrs;
		accessor.get_splvrs(splvrs);

		for (const auto& splvr : splvrs) {
			SpilloverityView::Ptr splw(new SpilloverityView(splvr.second, splvr.first));
			m_spilloveritys.push_back(std::move(splw));
		}
	}

	void LayeralityView::cellularityView(instance::ilayerality_host_accessor& accessor) {
		ImGui::SetNextItemOpen(true);
		if (!m_cellularitys.empty() && ImGui::TreeNode(("cellularity " + m_name).c_str())) {
			for (const auto& cellularity : m_cellularitys) {
				cellularity->view();
			}
			auto it = m_cellularitys.begin();
			while (it != m_cellularitys.end()) {
				if (it->get()->isShouldBeRemoved())
				{
					accessor.rm_cell(it->get()->name());
					it = m_cellularitys.erase(it);
				}
				else ++it;
			}
			ImGui::TreePop();
		}
	}

	void LayeralityView::spilloverityView(instance::ilayerality_host_accessor& accessor) {
		ImGui::SetNextItemOpen(true);
		if (!m_spilloveritys.empty() && ImGui::TreeNode(("spilloverity " + m_name).c_str())) {
			for (const auto& spilloverity : m_spilloveritys) {
				spilloverity->view();
			}
			auto it = m_spilloveritys.begin();
			while (it != m_spilloveritys.end()) {
				if (it->get()->isShouldBeRemoved())
				{
					accessor.rm_splvr(it->get()->name());
					it = m_spilloveritys.erase(it);
				}
				else ++it;
			}
			ImGui::TreePop();
		}
	}
	
	void LayeralityView::cellularityLoad(instance::ilayerality_host_accessor& accessor) {
		std::unordered_map<std::string, instance::icellularity_host_accessor&> cells;
		accessor.get_cells(cells);

		for (const auto& cell : cells) {
			CellularityView::Ptr sw(new CellularityView(cell.second, cell.first));
			m_cellularitys.push_back(std::move(sw));
		}
	}
	
	void LayeralityView::spilloverityLoad(instance::ilayerality_host_accessor& accessor) {
		std::unordered_map<std::string, instance::ispilloverity_host_accessor&> splvrs;
		accessor.get_splvrs(splvrs);

		for (const auto& splvr : splvrs) {
			SpilloverityView::Ptr sw(new SpilloverityView(splvr.second, splvr.first));
			m_spilloveritys.push_back(std::move(sw));
		}
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

			mSizeTypeInputedPopupBtn->display();
			ImGui::SameLine();
		}
		ImGui::PopStyleVar();

		mTreeNode->display();
	}
}