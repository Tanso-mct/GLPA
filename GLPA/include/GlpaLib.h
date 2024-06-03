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
    /// @brief Global instance for using Glpa library. No other instances can be created.
    static GlpaLib* instance;

    /// @brief Win main function argument.
    HINSTANCE hInstance;
    HINSTANCE hPrevInstance;
    LPSTR lpCmdLine;
    int nCmdShow;

    /// @brief Where to get messages from the Windows API.
    MSG msg;

    /// @brief Stores a pointer to a class that has Glpa base as its base class, one for each window created by the user.
    std::unordered_map<std::string, GlpaBase*> pBcs;

    /// @brief Variable to select each class from hwnd.
    std::unordered_map<HWND, std::string> bcHWnds;

    GlpaLib
    (
        const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, 
        const LPSTR arg_lpCmdLine, const int arg_nCmdShow
    );

public :
    ~GlpaLib();

    /// @brief When using Glpa lib, this function must be performed first.
    /// @param arg_hInstance Win main function argument.
    /// @param arg_hPrevInstance Win main function argument.
    /// @param arg_lpCmdLine Win main function argument.
    /// @param arg_nCmdShow Win main function argument.
    static void Start
    (
        const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, 
        const LPSTR arg_lpCmdLine, const int arg_nCmdShow
    );

    /// @brief Specify as the return value of the Win main function.
    static int Close();

    /// @brief Windows api window procedure functions. Receive and process messages in each window.
    /// @param hWnd Window handle.
    /// @param msg Message to retrieve in Windows API.
    /// @param wParam Additional information for Windows API messages.
    /// @param lParam Additional information for Windows API messages.
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

    static void CreateWindowNotApi(GlpaBase* pBc);
    static void ShowWindowNotApi(GlpaBase* pBc, int type);

    static void Load(GlpaBase* pBc);
    static void Release(GlpaBase* pBc);

    static void Run();
    

};

#endif GLPA_LIB_H_