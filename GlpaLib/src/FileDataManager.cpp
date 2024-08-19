#include "FileDataManager.h"
#include "GlpaLog.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Glpa::FileDataManager::FileDataManager()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
}

Glpa::FileDataManager::~FileDataManager()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
    for (auto& file : files)
    {
        files[file.first]->data->release();

        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete[" + file.first + "] data");
        delete files[file.first]->data;
        files[file.first]->data = nullptr;

        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete[" + file.first + "]");
        delete file.second;
    }

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Clear files");
    files.clear();
}

void Glpa::FileDataManager::newFile(std::string path)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "New file[" + path + "]");
    if (files.find(path) != files.end())
    {
        Glpa::outputErrorLog(__FILE__, __LINE__, "This applies to files that have already been added.");
        return;
    }

    Glpa::FileData* file = new Glpa::FileData();

    std::string name = path.substr(path.rfind("/") + 1, path.size() - path.rfind("/"));
    std::string extension 
    = name.substr(name.find(".") + 1, name.size() - name.find("."));

    if (extension == "png")
    {
        file->data = new Glpa::PngData();
    }
    else if (extension == "obj")
    {
        file->data = new Glpa::ObjData();
    }
    else
    {
        Glpa::runTimeError(__FILE__, __LINE__, "File format is not supported.");
    }

    file->data->filePath = path;
    file->data->fileName = name;
    file->data->extension = extension;

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Add file[" + path + "]");
    files.emplace(std::make_pair(path, file));
}

void Glpa::FileDataManager::deleteFile(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Release[" + path + "] data");
    files[path]->data->release();

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete[" + path + "] data");
    delete files[path]->data;
    files[path]->data = nullptr;

    files.erase(path);
}

void Glpa::FileDataManager::load(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Data[" + path + "]");
    files[path]->data->load();
}

void Glpa::FileDataManager::release(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Data[" + path + "]");
    files[path]->data->release();
}

std::string Glpa::FileDataManager::getFileName(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "File[" + path + "]");
    return files[path]->data->fileName;
}

int Glpa::FileDataManager::getWidth(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    if (Glpa::PngData* png = dynamic_cast<Glpa::PngData*>(files[path]->data))
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_FILE_DATA_MG, "Width[" + path + "]");
        return png->width;
    }
    else
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__,
            "It is not a file whose width can be obtained."
        );
    }
}

int Glpa::FileDataManager::getHeight(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    if (Glpa::PngData* png = dynamic_cast<Glpa::PngData*>(files[path]->data))
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_FILE_DATA_MG, "Height[" + path + "]");
        return png->height;
    }
    else
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__,
            "It is not a file whose height can be obtained."
        );
    }
}

int Glpa::FileDataManager::getChannels(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    if (Glpa::PngData* png = dynamic_cast<Glpa::PngData*>(files[path]->data))
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_FILE_DATA_MG, "Channels[" + path + "]");
        return png->channels;
    }
    else
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__,
            "It is not a file whose channels can be obtained."
        );
    }
}

LPDWORD Glpa::FileDataManager::getPngData(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    if (Glpa::PngData* png = dynamic_cast<Glpa::PngData*>(files[path]->data))
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_FILE_DATA_MG, "Data[" + path + "]");
        return png->getData();
    }
    else
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__,
            "It is not a file whose data can be obtained."
        );
    }
}

std::vector<Glpa::POLYGON> Glpa::FileDataManager::getPolyData(std::string path)
{
    if (Glpa::ObjData* obj = dynamic_cast<Glpa::ObjData*>(files[path]->data))
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_FILE_DATA_MG, "Data[" + path + "]");
        return obj->getPolyData();
    }
    else
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__,
            "It is not a file whose data can be obtained."
        );
    }
}

Glpa::RANGE_RECT Glpa::FileDataManager::getRangeRectData(std::string path)
{
    if (Glpa::ObjData* obj = dynamic_cast<Glpa::ObjData*>(files[path]->data))
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_FILE_DATA_MG, "Data[" + path + "]");
        return obj->getRangeRectData();
    }
    else
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__,
            "It is not a file whose data can be obtained."
        );
    }
}

void Glpa::PngData::load()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "PNG image[" + filePath + "]");
    stbi_uc* pixels = stbi_load(filePath.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels) {
        Glpa::runTimeError(__FILE__, __LINE__, "Failed to load PNG image.");
    }

    data = new DWORD[width * height];

    int pixelIndex = 0;

    for(UINT y = 0; y <= height; y++)
    {
        for(UINT x = 0; x <= width; x++)
        {
            if (x < width && y < height)
            {
                data[x+y*width] = (pixels[pixelIndex * 4 + 3] << 24) | 
                                  (pixels[pixelIndex * 4] << 16) | 
                                  (pixels[pixelIndex * 4 + 1] << 8) | 
                                  pixels[pixelIndex * 4 + 2];
                pixelIndex += 1;
            }
        }
    }

    stbi_image_free(pixels);
}

void Glpa::PngData::release()
{
    if (data == nullptr)
    {
        return;
    }

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "PNG image[" + filePath + "]");

    width = 0;
    height = 0;
    channels = 0;

    delete data;
    data = nullptr;
}

void Glpa::ObjData::load()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "OBJ file[" + filePath + "]");
    
    std::ifstream file(filePath);
    if (file.fail()) Glpa::runTimeError(__FILE__, __LINE__, "Failed to load OBJ file.");

    // Initialize data
    wv.clear();
    uv.clear();
    normal.clear();

    if (rangeRect != nullptr) delete rangeRect;
    rangeRect = new Glpa::RangeRect();

    if (polygons.size() != 0)
    {
        for (auto& polygon : polygons)
        {
            if (polygon == nullptr) continue;
            delete polygon;
        }
        polygons.clear();
    }


    std::string line;
    while (std::getline(file, line)) 
    {
        int space1 = line.find_first_of(" ");
        std::string type = line.substr(0, space1);
        std::string contents = line.substr(space1 + 1, line.size() - space1);

        if (type == "#") continue;
        else if (type == "") continue;
        else if (type == "mtllib") continue;
        else if (type == "g") continue;
        else if (type == "usemtl") continue;
        else if (type == "v")
        {
            Glpa::Vec3d* vec = new Glpa::Vec3d
            (
                std::stof(contents.substr(0, contents.find_first_of(" "))),
                std::stof(contents.substr(contents.find_first_of(" ") + 1, contents.find_last_of(" ") - contents.find_first_of(" "))),
                std::stof(contents.substr(contents.find_last_of(" ") + 1, contents.size() - contents.find_last_of(" ")))
            );

            wv.push_back(vec);
            rangeRect->addRangeV(vec);
        }
        else if (type == "vt")
        {
            Glpa::Vec2d* vec = new Glpa::Vec2d
            (
                std::stof(contents.substr(0, contents.find_first_of(" "))),
                std::stof(contents.substr(contents.find_first_of(" ") + 1, contents.size() - contents.find_first_of(" ")))
            );

            uv.push_back(vec);
        }
        else if (type == "vn")
        {
            Glpa::Vec3d* vec = new Glpa::Vec3d
            (
                std::stof(contents.substr(0, contents.find_first_of(" "))),
                std::stof(contents.substr(contents.find_first_of(" ") + 1, contents.find_last_of(" ") - contents.find_first_of(" "))),
                std::stof(contents.substr(contents.find_last_of(" ") + 1, contents.size() - contents.find_last_of(" ")))
            );

            normal.push_back(vec);
        }
        else if (type == "f")
        {
            std::vector<std::string> innerContents;
            innerContents.push_back(contents.substr(0, contents.find_first_of(" ")));
            innerContents.push_back(contents.substr(contents.find_first_of(" ") + 1, contents.find_last_of(" ") - contents.find_first_of(" ")));
            innerContents.push_back(contents.substr(contents.find_last_of(" ") + 1, contents.size() - contents.find_last_of(" ")));

            Glpa::Polygon* polygon = new Glpa::Polygon();
            for (int i = 0; i < innerContents.size(); i++)
            {
                std::string innerContent = innerContents[i];
                
                int vNum = std::stoi(innerContent.substr(0, innerContent.find_first_of("/")));
                int uvNum = std::stoi(innerContent.substr(innerContent.find_first_of("/") + 1, innerContent.find_last_of("/") - innerContent.find_first_of("/")));

                polygon->addV(vNum, uvNum);
            }

            int vnNum = std::stoi(innerContents[0].substr(innerContents[0].find_last_of("/") + 1, innerContents[0].size() - innerContents[0].find_last_of("/")));
            polygon->setNormal(*normal[vnNum - 1]);

            polygons.push_back(polygon);
        }
    }

    rangeRect->setWvs();

}

void Glpa::ObjData::release()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "OBJ file[" + filePath + "]");

    if (rangeRect != nullptr)
    {
        delete rangeRect;
        rangeRect = nullptr;
    }

    for (auto& vec : wv)
    {
        if (vec == nullptr) continue;
        delete vec;
    }
    wv.clear();

    for (auto& vec : uv)
    {
        if (vec == nullptr) continue;
        delete vec;
    }
    uv.clear();

    for (auto& vec : normal)
    {
        if (vec == nullptr) continue;
        delete vec;
    }
    normal.clear();

    for (auto& polygon : polygons)
    {
        if (polygon == nullptr) continue;
        delete polygon;
    }
    polygons.clear();
}

std::vector<Glpa::POLYGON> Glpa::ObjData::getPolyData()
{
    if (polygons.size() != 0)
    {
        std::vector<Glpa::POLYGON> polyData;
        for (auto& polygon : polygons)
        {
            polyData.push_back(polygon->getData(wv, uv));
        }

        return polyData;
    }
    else
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__,
            "There is no data to be returned."
        );
    }
}

Glpa::RANGE_RECT Glpa::ObjData::getRangeRectData()
{
    if (rangeRect != nullptr)
    {
        return rangeRect->getData();
    }
    else
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__,
            "There is no data to be returned."
        );
    }
}
