#pragma once
#include <vector>
#include <memory>

#include "../core/corelib.hpp"

struct GLFWwindow;

namespace Ui {
	class SubWindow
	{
	public:
		using Ptr = std::unique_ptr<SubWindow>;
		virtual void display() = 0;
	};

	class Application {
	public:
		Application();
		virtual ~Application();

		void run();

	private:
		GLFWwindow* m_glfwWindow = nullptr;
		std::vector<SubWindow::Ptr> m_subFrames;

		corelib* m_corelib = nullptr;

	};
}


