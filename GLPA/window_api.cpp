#include "window_api.h"

void WindowApi::storeWinMainArgument
(
    _In_ HINSTANCE arghInstance, _In_opt_ HINSTANCE arghPrevInstance, 
    _In_ LPSTR arglpCmdLine, _In_ int argnCmdShow
)
{
    hInstance = arghInstance;
    hPrevInstance = arghPrevInstance;
    lpCmdLine = arglpCmdLine;
    nCmdShow = argnCmdShow;
}

void WindowApi::registerClass
(
    
)
{

}