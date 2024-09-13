#include "Camera.cuh"
#include "GlpaLog.h"

Glpa::Camera::Camera(std::string argName, Glpa::Vec3d defPos, Glpa::Vec3d defRotate, float defFov, Glpa::Vec2d defAspectRatio, float defNearZ, float defFarZ)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    name = argName;
    pos = defPos;
    rotate = defRotate;
    fov = defFov;
    aspectRatio = defAspectRatio;
    nearZ = defNearZ;
    farZ = defFarZ;

    // Get screen size
    nearScrSize.x = std::fabs(std::tan(Glpa::RtoD(fov / 2)) * -nearZ) * 2;
    nearScrSize.y = std::fabs(nearScrSize.x * aspectRatio.y / aspectRatio.x);

    farScrSize.x = std::fabs(std::tan(Glpa::RtoD(fov / 2)) * -farZ) * 2;
    farScrSize.y = std::fabs(farScrSize.x * aspectRatio.y / aspectRatio.x);

    fovXzCos = cos(Glpa::RtoD(fov / 2));
    fovYzCos = std::fabs(-nearZ / sqrt(-nearZ*-nearZ + (nearScrSize.y/2) * (nearScrSize.y/2)));
}

Glpa::Camera::~Camera()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

Glpa::GPU_CAMERA Glpa::Camera::getData()
{
    Glpa::GPU_CAMERA camera;

    camera.pos.set(pos.x, pos.y, pos.z);
    camera.rotate.set(rotate.x, rotate.y, rotate.z);

    camera.fov = fov;
    camera.fovXzCos = fovXzCos;
    camera.fovYzCos = fovYzCos;

    camera.aspectRatio.set(aspectRatio.x, aspectRatio.y);

    camera.nearZ = nearZ;
    camera.farZ = farZ;

    camera.nearScrSize.set(nearScrSize.x, nearScrSize.y);

    float mtTransRot[16] =
    {
        std::cos(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.y)), 
        std::cos(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::sin(Glpa::RtoD(-rotate.x)) - std::sin(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.x)), 
        std::cos(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::cos(Glpa::RtoD(-rotate.x)) - std::sin(Glpa::RtoD(-rotate.z)) * -std::sin(Glpa::RtoD(-rotate.x)), 
        -pos.x,

        std::sin(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.y)),
        std::sin(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::sin(Glpa::RtoD(-rotate.x)) + std::cos(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.x)),
        std::sin(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::cos(Glpa::RtoD(-rotate.x)) + std::cos(Glpa::RtoD(-rotate.z)) * -std::sin(Glpa::RtoD(-rotate.x)),
        -pos.y,

        -std::sin(Glpa::RtoD(-rotate.y)),
        std::cos(Glpa::RtoD(-rotate.y)) * std::sin(Glpa::RtoD(-rotate.x)),
        std::cos(Glpa::RtoD(-rotate.y)) * std::cos(Glpa::RtoD(-rotate.x)),
        -pos.z,

        0, 0, 0, 1
    };
    camera.mtTransRot.set(mtTransRot);

    float mtRot[16] =
    {
        std::cos(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.y)), 
        std::cos(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::sin(Glpa::RtoD(-rotate.x)) - std::sin(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.x)), 
        std::cos(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::cos(Glpa::RtoD(-rotate.x)) - std::sin(Glpa::RtoD(-rotate.z)) * -std::sin(Glpa::RtoD(-rotate.x)), 
        0,

        std::sin(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.y)),
        std::sin(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::sin(Glpa::RtoD(-rotate.x)) + std::cos(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.x)),
        std::sin(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::cos(Glpa::RtoD(-rotate.x)) + std::cos(Glpa::RtoD(-rotate.z)) * -std::sin(Glpa::RtoD(-rotate.x)),
        0,

        -std::sin(Glpa::RtoD(-rotate.y)),
        std::cos(Glpa::RtoD(-rotate.y)) * std::sin(Glpa::RtoD(-rotate.x)),
        std::cos(Glpa::RtoD(-rotate.y)) * std::cos(Glpa::RtoD(-rotate.x)),
        0,

        0, 0, 0, 1
    };
    
    camera.mtRot.set(mtRot);

    // Defines the coordinates of the four vertices when the view volume is viewed from the positive y-axis direction.
    camera.vv.xzV[Glpa::VV_XZ_FACE::NEAR_LEFT].set(-nearScrSize.x / 2, -nearZ);
    camera.vv.xzV[Glpa::VV_XZ_FACE::FAR_LEFT].set(-farScrSize.x / 2, -farZ);
    camera.vv.xzV[Glpa::VV_XZ_FACE::FAR_RIGHT].set(farScrSize.x / 2, -farZ);
    camera.vv.xzV[Glpa::VV_XZ_FACE::NEAR_RIGHT].set(nearScrSize.x / 2, -nearZ);

    // Defines the coordinates of the four vertices when the view volume is viewed from the positive X-axis direction.
    camera.vv.yzV[Glpa::VV_YZ_FACE::NEAR_TOP].set(nearScrSize.y / 2, -nearZ);
    camera.vv.yzV[Glpa::VV_YZ_FACE::FAR_TOP].set(farScrSize.y / 2, -farZ);
    camera.vv.yzV[Glpa::VV_YZ_FACE::FAR_BOTTOM].set(-farScrSize.y / 2, -farZ);
    camera.vv.yzV[Glpa::VV_YZ_FACE::NEAR_BOTTOM].set(-nearScrSize.y / 2, -nearZ);


    // Defines the coordinates of the vertices in the camera 3D coordinates of the view volume.
    camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_LEFT].set(camera.vv.xzV[Glpa::VV_XZ_FACE::NEAR_LEFT].x, camera.vv.yzV[Glpa::VV_YZ_FACE::NEAR_TOP].x, -nearZ);
    camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_RIGHT].set(camera.vv.xzV[Glpa::VV_XZ_FACE::NEAR_RIGHT].x, camera.vv.yzV[Glpa::VV_YZ_FACE::NEAR_TOP].x, -nearZ);
    camera.vv.wv[Glpa::RECT_3D::FRONT_BOTTOM_RIGHT].set(camera.vv.xzV[Glpa::VV_XZ_FACE::NEAR_RIGHT].x, camera.vv.yzV[Glpa::VV_YZ_FACE::NEAR_BOTTOM].x, -nearZ);
    camera.vv.wv[Glpa::RECT_3D::FRONT_BOTTOM_LEFT].set(camera.vv.xzV[Glpa::VV_XZ_FACE::NEAR_LEFT].x, camera.vv.yzV[Glpa::VV_YZ_FACE::NEAR_BOTTOM].x, -nearZ);
    camera.vv.wv[Glpa::RECT_3D::BACK_TOP_LEFT].set(camera.vv.xzV[Glpa::VV_XZ_FACE::FAR_LEFT].x, camera.vv.yzV[Glpa::VV_YZ_FACE::FAR_TOP].x, -farZ);
    camera.vv.wv[Glpa::RECT_3D::BACK_TOP_RIGHT].set(camera.vv.xzV[Glpa::VV_XZ_FACE::FAR_RIGHT].x, camera.vv.yzV[Glpa::VV_YZ_FACE::FAR_TOP].x, -farZ);
    camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_RIGHT].set(camera.vv.xzV[Glpa::VV_XZ_FACE::FAR_RIGHT].x, camera.vv.yzV[Glpa::VV_YZ_FACE::FAR_BOTTOM].x, -farZ);
    camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_LEFT].set(camera.vv.xzV[Glpa::VV_XZ_FACE::FAR_LEFT].x, camera.vv.yzV[Glpa::VV_YZ_FACE::FAR_BOTTOM].x, -farZ);


    // Stores the coordinates of the vertices that are the start and end points of each line segment of the view volume.
    camera.vv.line[0].set(camera.vv.wv[Glpa::RECT_3D_LINE::L1_START], camera.vv.wv[Glpa::RECT_3D_LINE::L1_END]);
    camera.vv.line[1].set(camera.vv.wv[Glpa::RECT_3D_LINE::L2_START], camera.vv.wv[Glpa::RECT_3D_LINE::L2_END]);
    camera.vv.line[2].set(camera.vv.wv[Glpa::RECT_3D_LINE::L3_START], camera.vv.wv[Glpa::RECT_3D_LINE::L3_END]);
    camera.vv.line[3].set(camera.vv.wv[Glpa::RECT_3D_LINE::L4_START], camera.vv.wv[Glpa::RECT_3D_LINE::L4_END]);
    camera.vv.line[4].set(camera.vv.wv[Glpa::RECT_3D_LINE::L5_START], camera.vv.wv[Glpa::RECT_3D_LINE::L5_END]);
    camera.vv.line[5].set(camera.vv.wv[Glpa::RECT_3D_LINE::L6_START], camera.vv.wv[Glpa::RECT_3D_LINE::L6_END]);
    camera.vv.line[6].set(camera.vv.wv[Glpa::RECT_3D_LINE::L7_START], camera.vv.wv[Glpa::RECT_3D_LINE::L7_END]);
    camera.vv.line[7].set(camera.vv.wv[Glpa::RECT_3D_LINE::L8_START], camera.vv.wv[Glpa::RECT_3D_LINE::L8_END]);
    camera.vv.line[8].set(camera.vv.wv[Glpa::RECT_3D_LINE::L9_START], camera.vv.wv[Glpa::RECT_3D_LINE::L9_END]);
    camera.vv.line[9].set(camera.vv.wv[Glpa::RECT_3D_LINE::L10_START], camera.vv.wv[Glpa::RECT_3D_LINE::L10_END]);
    camera.vv.line[10].set(camera.vv.wv[Glpa::RECT_3D_LINE::L11_START], camera.vv.wv[Glpa::RECT_3D_LINE::L11_END]);
    camera.vv.line[11].set(camera.vv.wv[Glpa::RECT_3D_LINE::L12_START], camera.vv.wv[Glpa::RECT_3D_LINE::L12_END]);

    camera.vv.face[Glpa::FACE_3D::TOP].set
    (
        camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_LEFT], camera.vv.line[0].vec, camera.vv.line[4].vec
    );
    camera.vv.face[Glpa::FACE_3D::TOP].setSquare
    (
        camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_LEFT], camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_RIGHT], 
        camera.vv.wv[Glpa::RECT_3D::BACK_TOP_RIGHT], camera.vv.wv[Glpa::RECT_3D::BACK_TOP_LEFT]
    );

    camera.vv.face[Glpa::FACE_3D::FRONT].set
    (
        camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_LEFT], camera.vv.line[0].vec, camera.vv.line[1].vec
    );
    camera.vv.face[Glpa::FACE_3D::FRONT].setSquare
    (
        camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_LEFT], camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_RIGHT], 
        camera.vv.wv[Glpa::RECT_3D::FRONT_BOTTOM_RIGHT], camera.vv.wv[Glpa::RECT_3D::FRONT_BOTTOM_LEFT]
    );

    camera.vv.face[Glpa::FACE_3D::RIGHT].set
    (
        camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_RIGHT], camera.vv.line[1].vec, camera.vv.line[5].vec
    );
    camera.vv.face[Glpa::FACE_3D::RIGHT].setSquare
    (
        camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_RIGHT], camera.vv.wv[Glpa::RECT_3D::BACK_TOP_RIGHT], 
        camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_RIGHT], camera.vv.wv[Glpa::RECT_3D::FRONT_BOTTOM_RIGHT]
    );

    camera.vv.face[Glpa::FACE_3D::LEFT].set
    (
        camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_LEFT], camera.vv.line[3].vec, camera.vv.line[4].vec
    );
    camera.vv.face[Glpa::FACE_3D::LEFT].setSquare
    (
        camera.vv.wv[Glpa::RECT_3D::FRONT_TOP_LEFT], camera.vv.wv[Glpa::RECT_3D::BACK_TOP_LEFT], 
        camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_LEFT], camera.vv.wv[Glpa::RECT_3D::FRONT_BOTTOM_LEFT]
    );

    camera.vv.face[Glpa::FACE_3D::BACK].set
    (
        camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_RIGHT], camera.vv.line[8].vec, camera.vv.line[9].vec
    );
    camera.vv.face[Glpa::FACE_3D::BACK].setSquare
    (
        camera.vv.wv[Glpa::RECT_3D::BACK_TOP_LEFT], camera.vv.wv[Glpa::RECT_3D::BACK_TOP_RIGHT], 
        camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_RIGHT], camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_LEFT]
    );

    camera.vv.face[Glpa::FACE_3D::BOTTOM].set
    (
        camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_RIGHT], camera.vv.line[2].vec, camera.vv.line[6].vec
    );
    camera.vv.face[Glpa::FACE_3D::BOTTOM].setSquare
    (
        camera.vv.wv[Glpa::RECT_3D::FRONT_BOTTOM_LEFT], camera.vv.wv[Glpa::RECT_3D::FRONT_BOTTOM_RIGHT], 
        camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_RIGHT], camera.vv.wv[Glpa::RECT_3D::BACK_BOTTOM_LEFT]
    );

    return camera;
}

void Glpa::CAMERA_FACTORY::dFree(Glpa::GPU_CAMERA*& dCamera)
{
    if (!malloced) return;

    cudaFree(dCamera);
    dCamera = nullptr;
    malloced = false;
}

void Glpa::CAMERA_FACTORY::dMalloc(Glpa::GPU_CAMERA*& dCamera, Glpa::GPU_CAMERA &hCamera)
{
    if (malloced) dFree(dCamera);
    cudaMalloc(&dCamera, sizeof(Glpa::GPU_CAMERA));
    cudaMemcpy(dCamera, &hCamera, sizeof(Glpa::GPU_CAMERA), cudaMemcpyHostToDevice);

    malloced = true;
}

void Glpa::CAMERA_FACTORY::deviceToHost(Glpa::GPU_CAMERA*& dCamera, Glpa::GPU_CAMERA &hCamera)
{
    if (!malloced) Glpa::runTimeError(__FILE__, __LINE__, {"There is no memory on the camera device side."});
    cudaMemcpy(&hCamera, dCamera, sizeof(Glpa::GPU_CAMERA), cudaMemcpyDeviceToHost);
}
