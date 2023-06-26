#ifndef USERINPUT_H_
#define USERINPUT_H_

#include <windows.h>

// class UserInput
// {
//     //TODO:Think about what is the same process for both "WND LAU" and "WND PLAY".
//     public :
    
// };

class WndLAUInput
{
    public : 
    //Key message
    void keyDown(WPARAM wParam);
    // void keyUp(WPARAM wParam);

    // //Mouse Left button message
    // void mouseLbtnDown(LPARAM lParam);
    // void mouseLbtnUp(LPARAM lParam);
    // void mouseLbtnDblclick(LPARAM lParam);

    // //Mouse Right button message
    // void mouseRbtnDown(LPARAM lParam);
    // void mouseRbtnUp(LPARAM lParam);
    // void mouseRbtnDblClick(LPARAM lParam);

    // //Mouse Middle button message
    // void mouseMbtnDown(LPARAM lParam);
    // void mouseMbtnUp(LPARAM lParam);
    // void mouseMbtnWheel(LPARAM lParam);
};

class WndPLAYInput
{
    public : 
    //Key message
    void keyDown(WPARAM wParam);
    // void keyUp(WPARAM wParam);

    // //Mouse Left button message
    // void mouseLbtnDown(LPARAM lParam);
    // void mouseLbtnUp(LPARAM lParam);
    // void mouseLbtnDblclick(LPARAM lParam);

    // //Mouse Right button message
    // void mouseRbtnDown(LPARAM lParam);
    // void mouseRbtnUp(LPARAM lParam);
    // void mouseRbtnDblClick(LPARAM lParam);

    // //Mouse Middle button message
    // void mouseMbtnDown(LPARAM lParam);
    // void mouseMbtnUp(LPARAM lParam);
    // void mouseMbtnWheel(LPARAM lParam);
};

#endif USERINPUT_H_
