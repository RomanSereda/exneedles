#pragma once
#include "terminality.hpp"
#include "memory.cuh"

namespace instance {
	class LIBRARY_API host_terminality : public terminality<UPTR_TEMPLATE> {
	public:
		host_terminality(const ptree& root, const innate::layer& layer);
		ptree to_ptree() const;

		host_terminality(); //for test
	};


	class LIBRARY_API device_terminality : public terminality<PTR_TEMPLATE> {
	public:
		device_terminality(const ptree& root, const innate::layer& layer);
		~device_terminality();

	private:
		memory::const_empl::ptr m_const_cl = nullptr;
		memory::const_empl::ptr m_const_tr = nullptr;
	};

}
