#ifndef PLAYER_H_
#define PLAYER_H_

#include "file.h"
#include "cgmath.cuh"

#include "camera.h"
#include "hitbox.cuh"

typedef struct tagSTATUS_PLAYER
{
    double walkSpeed;
    double runSpeed;
    double jumpPower;
    double hp;
} STATUS_PLAYER;

class PLAYER
{
public :
    //TODO: Creating initialization functions
    VECTOR3D wPos;
    VECTOR3D rotAngle;
    VECTOR3D scaleRate;

    STATUS_PLAYER status;

    void initializeTrans();
    void initializeStatus();

    // Player object data
    OBJ_FILE head;
    OBJ_FILE body;

    // Hitbox
    HITBOX headBox;
    HITBOX bodyBox;

    void hitboxOperate();

    // Player Transform
    void posTrans();
    void rotTrans();
    void scaleTrans();

    // Camera Transform
    CAMERA cam;
    int cameraMode;
    void changeCameMode();


};

extern PLAYER deve001;



#endif PLAYER_H_
