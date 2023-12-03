#ifndef SYSTEM_H_
#define SYSTEM_H_

// RUN_STATUS
#define RUN_STATUS_LOAD 0
#define RUN_STATUS_MAINMENU 1
#define RUN_STATUS_ESCMENU 2
#define RUN_STATUS_PLAY1 3
#define RUN_STATUS_PLAY2 4
#define RUN_STATUS_PLAY3 5

class RUN_STATUS
{
public :
    int status = RUN_STATUS_LOAD;
};

extern RUN_STATUS mainRunSystem;


#endif SYSTEM_H_
