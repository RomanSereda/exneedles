#include "InnateSubWindow.hpp"

#include "TerminalityView.hpp"
#include "CellularityView.hpp"
#include "LayeralityView.hpp"

namespace Ui {
	InnateSubWindow::InnateSubWindow(instance::iregion& region)
		: mRegion(std::make_unique<RegionView>(region))
	{}

	void InnateSubWindow::display() {
		ImGui::Begin("innate");
		mRegion->view();
		ImGui::End();
	}

}