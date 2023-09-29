#include "InnateSubWindow.hpp"

#include "TerminalityView.hpp"
#include "CellularityView.hpp"
#include "LayeralityView.hpp"

namespace Ui {
	InnateSubWindow::InnateSubWindow() : mRegion(std::make_unique<RegionView>()){
	}

	void InnateSubWindow::display() {
		ImGui::Begin("innate");
		mRegion->view();
		ImGui::End();
	}

}