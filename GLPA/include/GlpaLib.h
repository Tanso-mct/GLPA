#ifndef GLPA_LIB_H_
#define GLPA_LIB_H_

#include <windows.h>
#include <vector>
#include <unordered_map>

#include "Macro.h"
#include "GlpaBase.h"

class GlpaLib
{
private :
    HINSTANCE hInstance;
    HINSTANCE hPrevInstance;
    LPSTR lpCmdLine;
    int nCmdShow;
    MSG msg;

    std::unordered_map<std::string, GlpaBase*> pBcs;

public :
    GlpaLib
    (
        const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, 
        const LPSTR arg_lpCmdLine, const int arg_nCmdShow
    );

    ~GlpaLib();

    void addBase(GlpaBase* pBc);
    void deleteBase(GlpaBase* pBc);

    void run();
    

};

#endif GLPA_LIB_H_