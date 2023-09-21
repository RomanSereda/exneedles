#pragma once
#include <vector>
#include <string>
#include "imgui.h"

namespace Ui {
	const ImVec4 ClearColor = ImColor(0.1f, 0.1f, 0.1f, 1.0f);

	struct ControlStyle {
		ImVec4 color = ImColor(0.2f, 0.2f, 0.2f, 1.0f);
		ImVec4 border = ImColor(0.5f, 0.5f, 0.5f, 1.0f);
		ImVec4 hovered = ImColor(0.3f, 0.3f, 0.3f, 1.0f);
		ImVec4 active = ImColor(0.4f, 0.4f, 0.4f, 1.0f);

		ImVec2 framePadding = { 3.0f, 2.0f };

		const int StyleColorCount = 4;
		const int StyleVarCount = 3;
	};
}
