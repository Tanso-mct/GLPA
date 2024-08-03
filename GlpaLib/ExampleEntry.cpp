#include "GlpaLib.h"
#include "ExampleBase.h"
#include "GlpaDebug.h"

int WINAPI WinMain
(
    const HINSTANCE hInstance, const HINSTANCE hPrevInstance,
    const LPSTR lpCmdLine, const int nCmdShow
){
    // Always do this first when using Glpa lib. Specify the argument of the win main function as the argument.
    GlpaLib::Start(hInstance, hPrevInstance, lpCmdLine, nCmdShow);

    // Create debug console
    Glpa::Debug* pDebugBase = new Glpa::Debug();
    pDebugBase->window->SetName(L"Debug Console");
    pDebugBase->window->SetApiClassName(L"debug_console");
    GlpaLib::AddBase(pDebugBase);
    GlpaLib::CreateWindowNotApi(pDebugBase);
    GlpaLib::ShowWindowNotApi(pDebugBase, SW_SHOWDEFAULT);
    GlpaLib::Load(pDebugBase);

    // Create an instance of a class that has the Glpa base class as its base class. Create windows and scenes in this class.
    ExampleBaseA* pBcA = new ExampleBaseA();
    pBcA->window->SetName(L"Example Base A");
    pBcA->window->SetApiClassName(L"example_base_a");
    pBcA->window->deleteViewStyle(WS_MAXIMIZEBOX);

    // Register the instance of the created class in glpa lib. This allows you to create windows and draw scenes.
    GlpaLib::AddBase(pBcA);

    // Create a window from the information set in the function of the created class instance.
    GlpaLib::CreateWindowNotApi(pBcA);
    // Display the created window. You can also change the display format.
    GlpaLib::ShowWindowNotApi(pBcA, SW_SHOWDEFAULT);

    // Load the first scene you set.
    GlpaLib::Load(pBcA);

    // Start of drawing loop. Drawing of the scene begins.
    GlpaLib::Run();

    // When using Glpa lib, the return value of the win main function must be as follows.
    return GlpaLib::Close();  
}