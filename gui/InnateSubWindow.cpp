#include "InnateSubWindow.hpp"

#include "TerminalityView.hpp"
#include "CellularityView.hpp"
#include "LayeralityView.hpp"

namespace Ui {
	InnateSubWindow::InnateSubWindow(instance::iregion_host_accessor& accessor)
		: mRegion(std::make_unique<RegionView>(accessor))
	{}

	void InnateSubWindow::display() {
		ImGui::Begin("innate");
		mRegion->view();
		ImGui::End();
	}

}