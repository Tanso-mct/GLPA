#include "GlpaLib.h"

#include "ExampleBase.h"
#include "ExampleScene3d.h"
#include "ExampleScene2d.h"

int WINAPI WinMain
(
    const HINSTANCE hInstance, const HINSTANCE hPrevInstance,
    const LPSTR lpCmdLine, const int nCmdShow
){
    GlpaLib::start(hInstance, hPrevInstance, lpCmdLine, nCmdShow);

    ExampleBase* pBc = new ExampleBase();



    MSG rtMsg = GlpaLib::getMsg();
    GlpaLib::close();
    return (int)rtMsg.wParam;  
}