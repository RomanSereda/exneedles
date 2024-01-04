#pragma once
#include <memory>
#include "Application.hpp"
#include "LayeralityView.hpp"

namespace Ui {
	class InnateSubWindow: public SubWindow {
	public:
		InnateSubWindow(instance::iregion_host_accessor& accessor);
		void display() override;

	private:
		RegionView::Ptr mRegion;
	};

}