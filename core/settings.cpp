#include "settings.hpp"
#include <windows.h>
#include <filesystem>

#include "assert.hpp"
#include "layerality.hpp"

static std::string get_exe_path()
{
	char buffer[MAX_PATH];
	GetModuleFileNameA(NULL, buffer, MAX_PATH);
	std::string path(buffer);
	return path.substr(0, path.find_last_of("\\"));
}

const std::string core::settings::setdir_name = "setdir";
const std::string core::settings::setdir_path = get_exe_path() + "\\" + core::settings::setdir_name;
const std::string core::settings::regions_path = setdir_path + "\\regions";

namespace core {
	settings& settings::instance() {
		static settings inst;
		inst.init();
		return inst;
	}

	void settings::save(const ptree& root, const std::string& name) const {
		boost::to_file(root, regions_path + "\\" + name);
	}

	ptree settings::read(const std::string& name) const {
		auto path = regions_path + "\\" + name;
		if (!std::filesystem::exists(path)) logexit();
		return boost::from_file(path);
	}

	void settings::init() {
		auto check_exist = [](const std::string& path) {
			if (!std::filesystem::exists(path))
				std::filesystem::create_directory(path);
		};

		check_exist(setdir_path);
		check_exist(regions_path);

		auto deffile = regions_path + "\\default";
		if (!std::filesystem::exists(deffile)) {
			innate::size size{1, 1};
			boost::to_file(boost::to_ptree(size), deffile);
		}
			
		if (std::filesystem::exists(regions_path))
			for (const auto& entry : std::filesystem::directory_iterator(regions_path)) {
				auto path = entry.path().string();
				auto name = path.substr(0, path.find_last_of("\\"));
				m_region_files.insert({ name, path });
			}
	}
}