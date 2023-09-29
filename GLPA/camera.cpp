#include "camera.h"

void CAMERA::initialize()
{
    
}

void CAMERA::defClippingArea()
{
    // Processing to be done only for the first execution
    if (!initialized)
    {
        viewPointA.resize(4);
        viewPointB.resize(4);

        initialized = true;
    }

    // define screen size
    nearScreenSize.width = tan(viewAngle / 2 * PI / 180) * nearZ * 2;
    nearScreenSize.height = nearScreenSize.width * aspectRatio.y / aspectRatio.x;

    farScreenSize.width = nearScreenSize.width / 2 * farZ / nearZ;
    farScreenSize.height = farScreenSize.width * aspectRatio.y / aspectRatio.x;
    
    // Define coordinates of view area vertices on xz axis
    viewPointA[VP1].x = nearScreenSize.width / 2 * -1;
    viewPointA[VP2].x = nearScreenSize.width / 2;
    viewPointA[VP3].x = farScreenSize.width / 2;
    viewPointA[VP4].x = farScreenSize.width / 2 * -1;

    viewPointA[VP1].z = nearZ;
    viewPointA[VP2].z = nearZ;
    viewPointA[VP3].z = farZ;
    viewPointA[VP4].z = farZ;

    // Define coordinates of view area vertices on yz axis
    viewPointB[VP1].y = nearScreenSize.height / 2;
    viewPointB[VP2].y = nearScreenSize.height / 2 * -1;
    viewPointB[VP3].y = farScreenSize.height / 2 * -1;
    viewPointB[VP4].y = farScreenSize.height / 2;

    viewPointB[VP1].z = nearZ;
    viewPointB[VP2].z = nearZ;
    viewPointB[VP3].z = farZ;
    viewPointB[VP4].z = farZ;




}