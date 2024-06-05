#include "GlpaLib.h"

#include "ExampleBase.h"
#include "ExampleScene3d.h"
#include "ExampleScene2d.h"

int WINAPI WinMain
(
    const HINSTANCE hInstance, const HINSTANCE hPrevInstance,
    const LPSTR lpCmdLine, const int nCmdShow
){
    // Always do this first when using Glpa lib. Specify the argument of the win main function as the argument.
    GlpaLib::Start(hInstance, hPrevInstance, lpCmdLine, nCmdShow);

    // Create an instance of a class that has the Glpa base class as its base class. Create windows and scenes in this class.
    ExampleBase* pBc = new ExampleBase();
    // Register the instance of the created class in glpa lib. This allows you to create windows and draw scenes.
    GlpaLib::AddBase(pBc);

    // Create a window from the information set in the function of the created class instance.
    GlpaLib::CreateWindowNotApi(pBc);
    // Display the created window. You can also change the display format.
    GlpaLib::ShowWindowNotApi(pBc, SW_SHOWDEFAULT);

    // Load the first scene you set.
    GlpaLib::Load(pBc);

    // Start of drawing loop. Drawing of the scene begins.
    GlpaLib::Run();

    // Delete the registered instance of the created class. 
    // Since the delete is also performed here, there is no need to perform the delete yourself within the win main function.
    GlpaLib::DeleteBase(pBc);

    // When using Glpa lib, the return value of the win main function must be as follows.
    return GlpaLib::Close();  
}