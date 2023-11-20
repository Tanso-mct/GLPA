#ifndef USERINPUT_H_
#define USERINPUT_H_

#include <tchar.h>
#include <windows.h>

#include "graphicbefore.h"
#include "camera.h"
#include "objectbefore.h"
#include "playerbefore.h"

// class UserInput
// {
//     //TODO:Think about what is the same process for both "WND LAU" and "WND PLAY".
//     public :
    
// };

class WndLAUInput
{
    public : 
    //Key message
    void keyDown(WPARAM w_Param);
    void keyUp(WPARAM w_Param);
    
    //Mouse message
    //Move message
    void mouseMove(LPARAM l_Param);
    
    //Left button message
    void mouseLbtnDown(LPARAM l_Param);
    // void mouseLbtnUp(LPARAM l_Param);
    // void mouseLbtnDblclick(LPARAM l_Param);

    //Right button message
    // void mouseRbtnDown(LPARAM l_Param);
    // void mouseRbtnUp(LPARAM l_Param);
    // void mouseRbtnDblClick(LPARAM l_Param);

    //Middle button message
    // void mouseMbtnDown(LPARAM l_Param);
    // void mouseMbtnUp(LPARAM l_Param);
    // void mouseMbtnWheel(LPARAM l_Param);
    
};

class WndPLAYInput
{
    public : 
    //Key message
    void keyDown(WPARAM w_Param);
    void keyUp(WPARAM w_Param);
    
    //Mouse message
    //Move message
    void mouseMove(LPARAM l_Param);
    
    //Left button message
    void mouseLbtnDown(LPARAM l_Param);
    // void mouseLbtnUp(LPARAM l_Param);
    // void mouseLbtnDblclick(LPARAM l_Param);

    //Right button message
    // void mouseRbtnDown(LPARAM l_Param);
    // void mouseRbtnUp(LPARAM l_Param);
    // void mouseRbtnDblClick(LPARAM l_Param);

    //Middle button message
    // void mouseMbtnDown(LPARAM l_Param);
    // void mouseMbtnUp(LPARAM l_Param);
    // void mouseMbtnWheel(LPARAM l_Param);
};

extern WndLAUInput UserInputWndLAU;
extern WndPLAYInput UserInputWndPLAY;


#endif USERINPUT_H_
