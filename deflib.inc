
#define DEBUG_PATH(X) "..//x64//Debug//"X".lib"
#define RELEASE_PATH(X) "..//x64//Release//"X".lib"

#ifdef _DEBUG                                      
#define PATH(X) DEBUG_PATH(X) 
#else                                              
#define PATH(X) RELEASE_PATH(X)  
#endif  

#define LIBRARY_COMMENT(X) comment(lib, PATH(X))


#ifdef _WINDLL                                     
	#define LIBRARY_API __declspec(dllexport)  
	#define EXPIMP_TEMPLATE
#else                                              
	#define LIBRARY_API __declspec(dllimport)     
	#define EXPIMP_TEMPLATE extern
	#pragma LIBRARY_COMMENT(LIBRARY)
#endif 