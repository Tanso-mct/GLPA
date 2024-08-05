#include "FileDataManager.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Glpa::FileDataManager::~FileDataManager()
{
    for (auto& file : files)
    {
        files[file.first]->data->release();
        delete files[file.first]->data;
        files[file.first]->data = nullptr;

        delete file.second;
    }

    files.clear();
}

void Glpa::FileDataManager::newFile(std::string path)
{
    if (files.find(path) != files.end())
    {
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

    files.emplace(std::make_pair(path, file));
    
}

void Glpa::FileDataManager::deleteFile(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    files[path]->data->release();

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

    files[path]->data->load();
}

void Glpa::FileDataManager::release(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    files[path]->data->release();
}

std::string Glpa::FileDataManager::getFileName(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

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

LPDWORD Glpa::FileDataManager::getData(std::string path)
{
    if (files.find(path) == files.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "This applies to files that have not been added.");
    }

    if (Glpa::PngData* png = dynamic_cast<Glpa::PngData*>(files[path]->data))
    {
        return png->data;
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

    width = 0;
    height = 0;
    channels = 0;

    delete data;
    data = nullptr;
}

void Glpa::ObjData::load()
{
}

void Glpa::ObjData::release()
{
}
