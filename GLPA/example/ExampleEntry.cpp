#include "GlpaLib.h"

#include "ExampleBase.h"
#include "ExampleScene3d.h"
#include "ExampleScene2d.h"
#include "ExampleInput.h"

int WINAPI WinMain
(
    const HINSTANCE hInstance, const HINSTANCE hPrevInstance,
    const LPSTR lpCmdLine, const int nCmdShow
){
    GlpaLib* pGlpa = new GlpaLib(hInstance, hPrevInstance, lpCmdLine, nCmdShow);

    ExampleBase* pBc = new ExampleBase();

}