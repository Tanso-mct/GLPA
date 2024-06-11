#include "Render.cuh"

Glpa::Render2d::Render2d()
{
}

Glpa::Render2d::~Render2d()
{
}

void Glpa::Render2d::run
(
    std::unordered_map<std::string, Glpa::SceneObject*> objs,
    std::map<int, std::vector<std::string>> drawOrder,
    HDC dc, LPDWORD buf
){
    // i = image amount
    // j = this image width x height

    // Separate processing depending on image or text.

    int totalImg = 0;
    int totalText = 0;

    std::vector<int> hImgPosX;
    std::vector<int> hImgPosY;
    std::vector<int> hImgSize;
    std::vector<LPDWORD> hImgData;

    for (auto& pair : drawOrder)
    {
        for (int i = 0; i < pair.second.size(); i++)
        {
            if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(objs[pair.second[i]]))
            {
                if (img->getVisible())
                {
                    Vec2d imgPos = img->getPos();
                    hImgPosX.push_back(imgPos.x);
                    hImgPosY.push_back(imgPos.y);
                    hImgSize.push_back(img->getWidth() * img->getHeight());

                    DWORD* dData;
                    size_t dataSize = imgPos.x * imgPos.y * sizeof(DWORD);
                    cudaMalloc(&dData, dataSize);
                    cudaMemcpy(dData, img->getData(), dataSize, cudaMemcpyHostToDevice);
                    hImgData.push_back(dData);
                }
            }

        }
    }

    int* dImgPosX, dImgPosY, dImgSize;
    LPDWORD* dImgData;

    cudaMalloc(&dImgData, hImgData.size() * sizeof(DWORD*));
    cudaMemcpy(d_ptrArray, h_ptrArray, numArrays * sizeof(DWORD*), cudaMemcpyHostToDevice);

}

Glpa::Render3d::Render3d()
{
}

Glpa::Render3d::~Render3d()
{
}

void Glpa::Render3d::run(std::unordered_map<std::string, Glpa::SceneObject*> objs, HDC dc, LPDWORD buf)
{

}
