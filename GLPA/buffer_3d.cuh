#ifndef BUFFER3D_H_
#define BUFFER3D_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>

#include "cg.h"
#include "error.h"

__global__ void glpaGpuDrawZBuff(
    double *d_z_buffer_comp,
    LPDWORD d_wnd_buffer,
    int scPixelSizeX,
    int scPixelSizeY,
    int scDpi);

class Buffer3d
{
public:
    void initialize(Vec2d screen_pixel_size, int screen_dpi);
    void drawZBuff(LPDWORD window_buffer, double *&z_buffer_comp);

private:
    Vec2d scPixelSize;
    int scDpi;
};

#endif BUFFER3D_H_
