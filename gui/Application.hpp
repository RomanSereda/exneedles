#pragma once
#include <vector>

struct GLFWwindow;

namespace Ui {
	class SubFrame
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
		std::vector<SubFrame*> m_subFrames;

	};

}


