#include "glpa.h"

void Glpa::initialize(HINSTANCE arghInstance, HINSTANCE arghPrevInstance, LPSTR arglpCmdLine, int argnCmdShow)
{
    hInstance = arghInstance;
    hPrevInstance = arghPrevInstance;
    lpCmdLine = arglpCmdLine;
    nCmdShow = argnCmdShow;
    ptWindowProc = windowProc;
}

void Glpa::createWindow(
    LPCWSTR wndName,
    LPCWSTR wndApiClassName,
    int wndWidth,
    int wndHeight,
    int wndDpi,
    double wndMaxFps,
    UINT wndStyle,
    LPWSTR loadIcon, 
    LPWSTR loadCursor,
    int backgroundColor,
    LPWSTR smallIcon,
    bool minimizeAuto,
    bool singleExistence
){

    Window tempWnd
    (
        wndName, wndApiClassName, wndWidth, wndHeight, wndDpi, wndMaxFps,
        wndStyle, loadIcon, loadCursor, backgroundColor, smallIcon, minimizeAuto, singleExistence
    );

    window[wndName] = tempWnd;
    window[wndName].create(hInstance, ptWindowProc);
    wndNames[window[wndName].hWnd] = wndName;
}

void Glpa::updateWindow(LPCWSTR wndName, int param){
    switch (param)
    {
    case GLPA_WINDOW_STATUS_DEF :
        window[wndName].updateStatus(GLPA_WINDOW_STATUS_DEF);
        break;

    case GLPA_WINDOW_STATUS_HIDE :
        window[wndName].updateStatus(GLPA_WINDOW_STATUS_HIDE);
        break;

    case GLPA_WINDOW_STATUS_MINIMIZE :
        window[wndName].updateStatus(GLPA_WINDOW_STATUS_MINIMIZE);
        break;

    default:
        throw std::runtime_error(ERROR_GLPA_UPDATE_WINDOW);
        break;
    }

}

void Glpa::setSingleWindow(bool single){
    if (single){
        singleWindow = true;
    }
    else{
        singleWindow = false;
    }
}

bool Glpa::dataSingleWindow(){
    return singleWindow;
}

void Glpa::runGraphicLoop(){
    while (true) {
        // Returns 1 (true) if a message is retrieved and 0 (false) if not.
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                OutputDebugStringW(_T("GLPA : EXIT\n"));

                // Exit from the loop when the exit message comes.
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } 

        for (auto& x: window) {
            if(x.second.isVisible()){
                x.second.graphicLoop();
                break;
            }
        }
    }
}

void Glpa::createScene(std::string scName, int selectType){
    scene.create(scName, selectType);
}

void Glpa::loadScene(std::string scName, LPCWSTR scFolderPath){
    WIN32_FIND_DATA findFileData;
    std::wstring wstrScFolderPath = scFolderPath;

    scene.setFolderPass(wstrScFolderPath);
    std::wstring wstrCutFolderPath = wstrScFolderPath;

    wstrScFolderPath += L"/*";

    HANDLE hFind = FindFirstFile(wstrScFolderPath.c_str(), &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(ERROR_GLPA_LOAD_SCENE);
    }

    std::vector<std::wstring> folderNames;
    std::vector<std::wstring> fileNames;

    std::unordered_map<std::wstring, std::vector<std::wstring>> allData;

    do {
        if ((findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
            wcscmp(findFileData.cFileName, L".") != 0 &&
            wcscmp(findFileData.cFileName, L"..") != 0
        ){
                folderNames.push_back(findFileData.cFileName);
                allData.emplace(findFileData.cFileName, fileNames);
        }
    } while (FindNextFile(hFind, &findFileData) != 0);

    std::wstring allFolderPass;
    for (auto it : folderNames){
        hFind = FindFirstFile((wstrCutFolderPath + L"/" + it + L"/*").c_str(), &findFileData);
        do {
            if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                allData[it].push_back(findFileData.cFileName);
            }
        } while (FindNextFile(hFind, &findFileData) != 0);
    }

    FindClose(hFind);

    scene.load(scName, wstrCutFolderPath, allData);
}


void Glpa::releaseScene(std::string scName){
    scene.release(scName);
}



void Glpa::selectUseScene(LPCWSTR targetWndName, std::string scName){
    window[targetWndName].setScene(&scene, scName);
}


Glpa glpa;

LRESULT CALLBACK windowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){
        switch (msg){
        case WM_SYSCOMMAND:
            if (wParam == SC_MINIMIZE) {
                for (auto& x: glpa.window) {
                    if(x.second.minimizeMsg(hWnd)){
                        return 0;
                    }
                }
            }

            return DefWindowProc(hWnd, msg, wParam, lParam);

        case WM_KILLFOCUS :
                for (auto& x: glpa.window) {
                    if(x.second.killFocusMsg(hWnd, glpa.dataSingleWindow())){
                        break;
                    }
                }
                return 0;

        case WM_SETFOCUS :
                for (auto& x: glpa.window) {
                    if(x.second.setFocusMsg(hWnd)){
                        break;
                    }
                }
                return 0;

        case WM_GETMINMAXINFO :
                for (auto& x: glpa.window) {
                    if(x.second.sizeMsg(hWnd, lParam)){
                        break;
                    }
                }
                return 0;

        case WM_CREATE :
                for (auto& x: glpa.window){
                    if (x.second.createMsg(hWnd)){
                        break;
                    }
                }
                return 0;

        case WM_PAINT :
                for (auto& x: glpa.window) {
                    if(x.second.paintMsg(hWnd)){
                        break;
                    }
                }
        
                return 0;

        case WM_CLOSE :
                for (auto& x: glpa.window) {
                    if(x.second.closeMsg(hWnd)){
                        break;
                    }
                }
                return 0;
                

        case WM_DESTROY :
                for (auto& x: glpa.window) {
                    if(x.second.destroyMsg(hWnd)){
                        break;
                    }
                }
                return 0;

        case WM_KEYDOWN :
                glpa.userInput.keyDown(hWnd, glpa.window[glpa.wndNames[hWnd]].useScene, wParam, lParam);
                return 0;

        case WM_KEYUP :
        case WM_LBUTTONDOWN :
        case WM_LBUTTONUP :
        case WM_LBUTTONDBLCLK :
        case WM_RBUTTONDOWN :
        case WM_RBUTTONUP :
        case WM_RBUTTONDBLCLK :
        case WM_MBUTTONDOWN :
        case WM_MBUTTONDBLCLK :
        case WM_MBUTTONUP :
        case WM_MOUSEWHEEL :
        case WM_MOUSEMOVE :
                for (auto& x: glpa.window) {
                    if(x.second.userMsg(hWnd)){
                        break;
                    }
                }
                return 0;



        default :
                return DefWindowProc(hWnd, msg, wParam, lParam);
        }
        return 0;
}