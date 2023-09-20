#pragma once
#include <memory>
#include "Application.hpp"
#include "LayeralityView.hpp"

namespace Ui {
	class InnateSubWindow: public SubWindow {
	public:
		InnateSubWindow();
		void display() override;

	private:
		RegionView::Ptr mRegion;
	};

}