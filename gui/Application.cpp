#include "Application.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glfw/glfw3.h>
#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "OpenGL32.lib")

#include "assert.hpp"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

static void glfw_error_callback(int error, const char* description) {
	fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

#include "Style.hpp"
#include "InnateSubWindow.hpp"

namespace Ui {
	Application::Application() {
        m_corelib = new corelib();

		glfwSetErrorCallback(glfw_error_callback);
		if (!glfwInit()) logexit();

		const char* glsl_version = "#version 130";
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

        m_glfwWindow = glfwCreateWindow(640, 480, "ExNeedles Project", nullptr, nullptr);
        if (m_glfwWindow == nullptr) logexit();
     
        glfwMakeContextCurrent(m_glfwWindow);
        glfwSwapInterval(1);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(m_glfwWindow, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        auto innateSubWindow = new InnateSubWindow(m_corelib->system().accessor());

        m_subFrames.push_back(SubWindow::Ptr(innateSubWindow));
	}

	Application::~Application() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(m_glfwWindow);
        glfwTerminate();

        delete m_corelib;
	}

	void Application::run() {
        while (!glfwWindowShouldClose(m_glfwWindow)) {
            glfwPollEvents();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // Windows
            for (auto& subFrame : m_subFrames) {
                subFrame->display();
            }

            // Rendering
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(m_glfwWindow, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(ClearColor.x, ClearColor.y, ClearColor.z, ClearColor.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(m_glfwWindow);
        }
	}
}