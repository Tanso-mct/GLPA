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
    static HINSTANCE hInstance;
    static HINSTANCE hPrevInstance;
    static LPSTR lpCmdLine;
    static int nCmdShow;
    static MSG msg;

    static std::unordered_map<std::string, GlpaBase*> pBcs;
    static std::unordered_map<HWND, std::string> bcHWnds;

public :
    GlpaLib
    (
        const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, 
        const LPSTR arg_lpCmdLine, const int arg_nCmdShow
    );

    ~GlpaLib();

    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

    void addBase(GlpaBase* pBc);
    void deleteBase(GlpaBase* pBc);

    void run();
    

};

#endif GLPA_LIB_H_