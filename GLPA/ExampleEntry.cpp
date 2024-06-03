#include "GlpaLib.h"

#include "ExampleBase.h"
#include "ExampleScene3d.h"
#include "ExampleScene2d.h"

int WINAPI WinMain
(
    const HINSTANCE hInstance, const HINSTANCE hPrevInstance,
    const LPSTR lpCmdLine, const int nCmdShow
){
    GlpaLib::Start(hInstance, hPrevInstance, lpCmdLine, nCmdShow);

    ExampleBase* pBc = new ExampleBase();

    GlpaLib::AddBase(pBc);

    GlpaLib::CreateWindowNotApi(pBc);
    GlpaLib::ShowWindowNotApi(pBc, SW_SHOWDEFAULT);

    GlpaLib::Load(pBc);

    GlpaLib::Run();

    return GlpaLib::Close();  
}