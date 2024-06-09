#ifndef GLPA_LIB_H_
#define GLPA_LIB_H_

#include <windows.h>
#include <vector>
#include <unordered_map>

#include "Constant.h"
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

    std::vector<GlpaBase*> pBcs;

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


    /// @brief Processing to be performed when a window minimize message is received.
    /// @param bc A class that identifies which glpa base class has the base class from the window handle.
    void minimizeMsg(GlpaBase* bc);

    /// @brief The message is processed when the window is resized and prevents the window from being resized by the user.
    /// @param bc A class that identifies which glpa base class has the base class from the window handle.
    void editSizeMsg(GlpaBase* bc, LPARAM lParam);

    /// @brief A create message may be received after the window is created. Create a DC.
    /// @param bc A class that identifies which glpa base class has the base class from the window handle.
    void createMsg(GlpaBase* bc);

    /// @brief A redraw message is sent, and the message is received and processed.
    /// @param bc A class that identifies which glpa base class has the base class from the window handle.
    void paintMsg(GlpaBase* bc);

    /// @brief Receive and process messages that occur when a window is closed. 
    /// It also releases the loaded data of instances of classes that have the glpa base class as their base class.
    /// @param bc A class that identifies which glpa base class has the base class from the window handle.
    void closeMsg(GlpaBase* bc);

    /// @brief Delete the instance of the class whose base class is the glpa base class that is assigned to the closed window. 
    /// If all windows are closed, exit the message loop.
    /// @param bc A class that identifies which glpa base class has the base class from the window handle.
    void destroyMsg(GlpaBase* bc);


    /// @brief Take action when a key down message is received.
    /// @param bc A class that identifies which glpa base class has the base class from the window handle.
    void keyDownMsg(GlpaBase* bc, UINT msg, WPARAM wParam, LPARAM lParam);

    /// @brief Perform processing when a keyup message is received.
    /// @param bc A class that identifies which glpa base class has the base class from the window handle.
    void keyUpMsg(GlpaBase* bc, UINT msg, WPARAM wParam, LPARAM lParam);

    /// @brief Process when a mouse input message is received.
    /// @param bc A class that identifies which glpa base class has the base class from the window handle.
    void mouseMsg(GlpaBase* bc, UINT msg, WPARAM wParam, LPARAM lParam);


    /// @brief Add an instance of a class that has the GlpaBase class as its base class. 
    /// This must be done in the user's class or in the WinMain function.
    /// @param pBc A pointer to a class instance whose base class is the Glpa base class.
    static void AddBase(GlpaBase* pBc);

    /// @brief Delete instances of classes whose base class is the already added GlpaBase class.
    /// @param pBc A pointer to a class instance whose base class is the Glpa base class.
    static void DeleteBase(GlpaBase* pBc);


    /// @brief Create a window from an instance of a class that has the Glpa base class as its base class.
    /// @param pBc A pointer to a class instance whose base class is the Glpa base class.
    static void CreateWindowNotApi(GlpaBase* pBc);

    /// @brief Change the window display format.
    /// @param pBc A pointer to a class instance whose base class is the Glpa base class.
    /// @param type Specifies the display format of the window starting with sw in the Windows API.
    static void ShowWindowNotApi(GlpaBase* pBc, int type);
    

    /// @brief Load a class that has the Glpa base class as its base class.
    /// @param pBc A pointer to a class instance whose base class is the Glpa base class.
    static void Load(GlpaBase* pBc);

    /// @brief Release loaded classes whose base class is the already loaded glpa base class.
    /// @param pBc A pointer to a class instance whose base class is the Glpa base class.
    static void Release(GlpaBase* pBc);

    /// @brief Starts the message loop and begins graphics processing.
    static void Run();
    

};

#endif GLPA_LIB_H_