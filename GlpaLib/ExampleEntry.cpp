#include "GlpaLib.h"
#include "GlpaConsole.h"

#include "ExampleBase.h"

int WINAPI WinMain
(
    const HINSTANCE hInstance, const HINSTANCE hPrevInstance,
    const LPSTR lpCmdLine, const int nCmdShow
){
    // Always do this first when using Glpa lib. Specify the argument of the win main function as the argument.
    GlpaLib::Start(hInstance, hPrevInstance, lpCmdLine, nCmdShow, true);

    // Start of drawing loop. Drawing of the scene begins.
    GlpaLib::Run();

    // When using Glpa lib, the return value of the win main function must be as follows.
    return GlpaLib::Close();  
}