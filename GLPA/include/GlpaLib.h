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
    static GlpaLib* instance;

    HINSTANCE hInstance;
    HINSTANCE hPrevInstance;
    LPSTR lpCmdLine;
    int nCmdShow;
    MSG msg;

    std::unordered_map<std::string, GlpaBase*> pBcs;
    std::unordered_map<HWND, std::string> bcHWnds;

public :
    static void Start
    (
        const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, 
        const LPSTR arg_lpCmdLine, const int arg_nCmdShow
    );
    static void Close();

    GlpaLib
    (
        const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, 
        const LPSTR arg_lpCmdLine, const int arg_nCmdShow
    );

    ~GlpaLib();

    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

    void minimizeMsg(GlpaBase* bc);
    void editSizeMsg(GlpaBase* bc, LPARAM lParam);
    void createMsg(GlpaBase* bc);
    void paintMsg(GlpaBase* bc);
    void closeMsg(GlpaBase* bc);
    void destroyMsg(GlpaBase* bc);

    void keyDownMsg(GlpaBase* bc, UINT msg, WPARAM wParam, LPARAM lParam);
    void keyUpMsg(GlpaBase* bc, UINT msg, WPARAM wParam, LPARAM lParam);
    void mouseMsg(GlpaBase* bc, UINT msg, WPARAM wParam, LPARAM lParam);

    static void AddBase(GlpaBase* pBc);
    static void DeleteBase(GlpaBase* pBc);

    static MSG getMsg(){return instance->msg;}

    static void CreateWindowNotApi(GlpaBase* pBc);
    static void ShowWindowNotApi(GlpaBase* pBc, int type);

    static void Load(GlpaBase* pBc);
    static void Release(GlpaBase* pBc);

    static void Run();
    

};

#endif GLPA_LIB_H_