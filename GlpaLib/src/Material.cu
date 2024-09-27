#include "Material.cuh"
#include "ErrorHandler.cuh"

Glpa::Material::Material(std::string argName, std::string argBaseColorFilePath)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    name = argName;
    diffuseFilePath = argBaseColorFilePath;
    
}

Glpa::Material::~Material()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

void Glpa::Material::setManager(Glpa::FileDataManager *argFileDataManager)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + name + "]");
    fileDataManager = argFileDataManager;
}

int Glpa::Material::GetMtWidth(std::string mtName)
{
    if (mtName == Glpa::MATERIAL_DIFFUSE) return fileDataManager->getWidth(diffuseFilePath);
    else if (mtName == Glpa::MATERIAL_ORM) return fileDataManager->getWidth(ormFilePath);
    else if (mtName == Glpa::MATERIAL_NORMAL) return fileDataManager->getWidth(normalFilePath);
    else
    {
        Glpa::runTimeError(__FILE__, __LINE__, {mtName + " is not a valid material name"});
        return -1;
    };
}

int Glpa::Material::GetMtHeight(std::string mtName)
{
    if (mtName == Glpa::MATERIAL_DIFFUSE) return fileDataManager->getHeight(diffuseFilePath);
    else if (mtName == Glpa::MATERIAL_ORM) return fileDataManager->getHeight(ormFilePath);
    else if (mtName == Glpa::MATERIAL_NORMAL) return fileDataManager->getHeight(normalFilePath);
    else
    {
        Glpa::runTimeError(__FILE__, __LINE__, {mtName + " is not a valid material name"});
        return -1;
    };
}

LPDWORD Glpa::Material::GetMtData(std::string mtName)
{
    if (mtName == Glpa::MATERIAL_DIFFUSE) return fileDataManager->getPngData(diffuseFilePath);
    else if (mtName == Glpa::MATERIAL_ORM) return fileDataManager->getPngData(ormFilePath);
    else if (mtName == Glpa::MATERIAL_NORMAL) return fileDataManager->getPngData(normalFilePath);
    else
    {
        Glpa::runTimeError(__FILE__, __LINE__, {mtName + " is not a valid material name"});
        return LPDWORD();
    };
}

Glpa::GPU_MATERIAL Glpa::Material::getData()
{
    Glpa::GPU_MATERIAL material;

    material.baseColor = fileDataManager->getPngData(diffuseFilePath);

    return material;
}

void Glpa::Material::load()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + name + "]");
    loaded = true;

    fileDataManager->newFile(diffuseFilePath);
    fileDataManager->load(diffuseFilePath);
}

void Glpa::Material::release()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + name + "]");
    loaded = false;

    fileDataManager->release(diffuseFilePath);
}

void Glpa::MATERIAL_FACTORY::dFree(Glpa::GPU_MATERIAL*& dMts)
{
    if (!malloced) return;

    for (size_t i = 0; i < idMap.size(); i++)
    {
        LPDWORD dBaseColor = nullptr;
        cudaMemcpy(dBaseColor, &dMts[i].baseColor, sizeof(LPDWORD), cudaMemcpyDeviceToHost);
        cudaFree(dBaseColor);
    }

    cudaFree(dMts);
    dMts = nullptr;

    idMap.clear();

    malloced = false;
}

void Glpa::MATERIAL_FACTORY::dMalloc(Glpa::GPU_MATERIAL*& dMts, std::unordered_map<std::string, Glpa::Material *> &sMts)
{
    std::vector<Glpa::GPU_MATERIAL> hMts;
    int id = 0;
    for (auto& pair : sMts)
    {
        Glpa::GPU_MATERIAL mt;
        cudaMalloc
        (
            &mt.baseColor, 
            pair.second->GetMtWidth(Glpa::MATERIAL_DIFFUSE) * pair.second->GetMtHeight(Glpa::MATERIAL_DIFFUSE) * sizeof(DWORD)
        );

        cudaMemcpy
        (
            mt.baseColor, 
            pair.second->GetMtData(Glpa::MATERIAL_DIFFUSE), 
            pair.second->GetMtWidth(Glpa::MATERIAL_DIFFUSE) * pair.second->GetMtHeight(Glpa::MATERIAL_DIFFUSE) * sizeof(DWORD), 
            cudaMemcpyHostToDevice
        );

        idMap[pair.first] = id;

        hMts.push_back(mt);
        id++;
    }

    cudaMalloc(&dMts, sMts.size() * sizeof(Glpa::GPU_MATERIAL));
    cudaMemcpy(dMts, hMts.data(), sMts.size() * sizeof(Glpa::GPU_MATERIAL), cudaMemcpyHostToDevice);

    for (int i = 0; i < hMts.size(); i++)
    {
        cudaFree(hMts[i].baseColor);
    }

    malloced = true;
}
