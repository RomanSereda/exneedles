#pragma once
#include <map>
#include "boost.hpp"

namespace core {
	class settings {
	public:
		static const std::string setdir_name;
		static const std::string setdir_path;
		static const std::string regions_path;

		static settings& instance();

		void save(const ptree& root, const std::string& name = "default") const;
		ptree read(const std::string& name = "default") const;

	private:
		void init();
		std::map<std::string, std::string> m_region_files;
	};
}