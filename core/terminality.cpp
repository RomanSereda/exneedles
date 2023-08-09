#include "terminality.hpp"

namespace instance {
	int cluster::terminal_bytes_size() const {
		int size = -1;
		data::terminal::foreach(std::get<__const__ innate::terminal*>(innate), [&size](auto* p) {
			size = sizeof(*p);
			return true;
		});
		return size;
	}

	ptree cluster::operator=(instance::cluster&) {
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
}