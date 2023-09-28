#include "camera.h"

void CAMERA::initialize()
{
    
}

void CAMERA::defClippingArea()
{
    if (!initialized)
    {
        viewPointA.resize(4);
        viewPointB.resize(4);

        initialized = true;
    }

    // viewPointA[0].x = tan()


}