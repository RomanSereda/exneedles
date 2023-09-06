#pragma once
#include <vector>

struct GLFWwindow;

namespace Ui {
	class SubWindow
	{
	public:
		virtual void display() = 0;
	};

	class Application {
	public:
		Application();
		virtual ~Application();

		void run();

	private:
		GLFWwindow* m_glfwWindow = nullptr;
		std::vector<SubWindow*> m_subFrames;

	};
}


