#pragma once
#ifdef CORE_EXPORTS                               
#define LIB_EXPORTS  
#else
#define LIBRARY "core"
#endif

#include "../deflib.inc"
#include "../core/layerality.hpp"

namespace core
{
	class LIBRARY_API device {
	public:
		device();
		~device();
	};
}

namespace instance {
	class LIBRARY_API cluster : protected data::cluster {
	public:
		int terminal_bytes_size() const;

		ptree operator=(instance::cluster&) {
			auto cl = std::get<__const__ innate::cluster*>(innate);
			auto tr = std::get<__const__ innate::terminal*>(innate);

			/*std::ostringstream archive_stream;
			boost::archive::text_oarchive archive(archive_stream);
			archive << cl;
			archive << tr;

			root.add(cNameJsonKey, archive_stream.str());*/

			auto root = boost::to_ptree(cl);

			return root;
		}

		/*cluster& operator=(const ptree& root) {



			return *this;
		}*/

	private:
		const std::string cNameJsonKey = "cluster";
	};
}

