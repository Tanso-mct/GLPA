// #define DEBUG_FILE_
#include "file.h"

int FILELOAD::loadBinary(int fileType, std::string inputFileName)
{
    std::string filePath ("../loadfiles/");
    fileName = inputFileName;

    switch (fileType)
    {
    case FILETYPE_BMP :
        filePath.append("bmp");
        break;
    
    case FILETYPE_PNG :
        filePath.append("png");
        break;
    
    default:
        break;
    }

    filePath.append("/");
    filePath.append(fileName);
    
    // Read in a binary file from a filename
    std::ifstream file(filePath, std::ios::binary);
    if (file.fail())
    {
        #ifdef DEBUG_FILE_

        OutputDebugStringA("binary file open failed\n");

        #endif

        loadStatus = LOAD_FAILURE;
        return 1;
    }

    loadStatus = STANDBY_LOAD;

    // Check the read size
    file.seekg(0, std::ios::end);
    long long int size = file.tellg();
    file.seekg(0);
    std::vector<std::string> tempBinaryData (size);

    // Output read data to char type
    char *fileData = new char[size];
    file.read(fileData, size);

    #ifdef DEBUG_FILE_

    // Output size
    OutputDebugStringA(("size = " + std::to_string(size) + "\n").c_str());

    #endif p

    char hexChar[9];
    int decimal;

    // Storing Binary Data
    loadStatus = LOADING;
    for (int i = 1; i < size + 1; i++)
    {
        decimal = std::stoi(std::to_string(fileData[i - 1]));
        sprintf_s(hexChar, sizeof(hexChar),"%.2X", decimal);
        std::string hexString(hexChar);
        if (std::stoi(std::to_string(fileData[i - 1])) < 0)
        {
            hexString.erase(0, 6);
            tempBinaryData[i - 1] = hexString;
        }
        else
        {
            tempBinaryData[i - 1] = hexString;
        }
    }
    loadStatus = ENDED_PROCESS;
    
    binaryData.swap(tempBinaryData);

    delete fileData;

    #ifdef DEBUG_FILE_

    OutputDebugStringA("END");

    #endif

    loadStatus = LOAD_SUCCESS;
    return 0;
}

void FILELOAD::checkBinary()
{
    // Display hexadecimal binary data
    for (int i = 1; i < binaryData.size() + 1; i++)
    {
        OutputDebugStringA("   ");
        OutputDebugStringA((binaryData[i - 1]).c_str());

        if ((i % 16) == 0)
        {
            OutputDebugStringA("\n");
        }
    }
}

int BMP_FILE::readBinary()
{
    return 0;
}

int OBJ_FILE::loadData(std::string inputFileName)
{
    std::string filePath ("../loadfiles/");
    fileName = inputFileName;

    filePath.append("obj");
    filePath.append("/");
    filePath.append(fileName);

    std::ifstream file(filePath);

    if (file.fail())
    {
        #ifdef DEBUG_FILE_

        OutputDebugStringA("obj file open failed\n");

        #endif
        loadStatus = LOAD_FAILURE;
        return 1;
    }

    // Initialize data
    range.status = false;

    v.world.resize(0);
    v.uv.resize(0);
    v.normal.resize(0);

    calcV.world.resize(0);
    calcV.uv.resize(0);
    calcV.normal.resize(0);

    poly.v.resize(0);
    poly.uv.resize(0);
    poly.normal.resize(0);

    loadStatus = STANDBY_LOAD;

    std::string tag;
    std::string line;
    std::string name;
    std::size_t punc1;
    std::size_t punc2;
    std::size_t punc3;
    std::size_t punc4;
    VECTOR3D num3d;
    VECTOR2D num2d;
    NUMCOMB3 numComb3;

    loadStatus = LOADING;

    while (std::getline(file, line)) {
        punc1 = line.find(" ");
        tag = line.substr(0, punc1);

        // Branching by TAG
        if (tag == "v")
        {
            // Save the first number
            punc2 = line.find(" ", tag.size() + 2);
            num3d.x = std::stod(line.substr(tag.size() + 1, punc2 - (tag.size() + 1)));

            // Save the second number
            punc3 = line.find(" ", punc2 + 1);
            num3d.y = std::stod(line.substr(punc2 + 1, punc3 - (punc2 + 1)));
            
            // Save the third number
            num3d.z = std::stod(line.substr(punc3 + 1, line.size() - (punc3 + 1)));

            if (range.status)
            {
                // Processing with respect to origin point
                if (num3d.x < range.origin.x)
                {
                    range.origin.x = num3d.x;
                }
                if (num3d.y < range.origin.y)
                {
                    range.origin.y = num3d.y;
                }
                if (num3d.z > range.origin.z)
                {
                    range.origin.z = num3d.z;
                }

                // Processing with respect to opposite point
                if (num3d.x > range.opposite.x)
                {
                    range.opposite.x = num3d.x;
                }
                if (num3d.y > range.opposite.y)
                {
                    range.opposite.y = num3d.y;
                }
                if (num3d.z < range.opposite.z)
                {
                    range.opposite.z = num3d.z;
                }
            }
            else
            {
                range.origin.x = num3d.x;
                range.origin.y = num3d.y;
                range.origin.z = num3d.z;

                range.opposite.x = num3d.x;
                range.opposite.y = num3d.y;
                range.opposite.z = num3d.z;
                range.status = true;
            }

            v.world.push_back(num3d);
        }
        else if (tag == "vt")
        {
            // Save the first number
            punc2 = line.find(" ", tag.size() + 2);
            num2d.x = std::stod(line.substr(tag.size() + 1, punc2 - (tag.size() + 1)));

            // Save the second number
            punc3 = line.find(" ", punc2 + 1);
            num2d.y = std::stod(line.substr(punc2 + 1, line.size() - (punc2 + 1)));

            v.uv.push_back(num2d);
        }
        else if (tag == "vn")
        {
            // Save the first number
            punc2 = line.find(" ", tag.size() + 2);
            num3d.x = std::stod(line.substr(tag.size() + 1, punc2 - (tag.size() + 1)));

            // Save the second number
            punc3 = line.find(" ", punc2 + 1);
            num3d.y = std::stod(line.substr(punc2 + 1, punc3 - (punc2 + 1)));
            
            // Save the third number
            num3d.z = std::stod(line.substr(punc3 + 1, line.size() - (punc3 + 1)));

            v.normal.push_back(num3d);
        }
        else if (tag == "f")
        {
            // Save vertex numbers
            // Save the first number
            punc2 = line.find("/", tag.size() + 2);
            numComb3.num1 = std::stoi(line.substr(tag.size() + 1, punc2 - (tag.size() + 1)));

            // Save the second number
            punc3 = line.find("/", punc2 + 1);
            numComb3.num2 = std::stoi(line.substr(punc2 + 1, punc3 - (punc2 + 1)));

            // Save the third number
            punc4 = line.find(" ", punc3 + 1);
            numComb3.num3 = std::stoi(line.substr(punc3 + 1, punc4 - (punc3 + 1)));

            poly.v.push_back(numComb3);

            // Save uv numbers
            // Save the first number
            punc2 = line.find("/", punc4 + 1);
            numComb3.num1 = std::stoi(line.substr(punc4 + 1, punc2 - (punc4 + 1)));

            // Save the second number
            punc3 = line.find("/", punc2 + 1);
            numComb3.num2 = std::stoi(line.substr(punc2 + 1, punc3 - (punc2 + 1)));

            // Save the third number
            punc4 = line.find(" ", punc3 + 1);
            numComb3.num3 = std::stoi(line.substr(punc3 + 1, punc4 - (punc3 + 1)));

            poly.uv.push_back(numComb3);

            // Save normal numbers
            // Save the first number
            punc2 = line.find("/", punc4 + 1);
            numComb3.num1 = std::stoi(line.substr(punc4 + 1, punc2 - (punc4 + 1)));

            // Save the second number
            punc3 = line.find("/", punc2 + 1);
            numComb3.num2 = std::stoi(line.substr(punc2 + 1, punc3 - (punc2 + 1)));

            // Save the third number
            numComb3.num3 = std::stoi(line.substr(punc3 + 1, line.size() - (punc3 + 1)));

            poly.normal.push_back(numComb3);
        }
    }

    range.wVertex.resize(8);

    range.wVertex[0] = {range.origin.x, range.opposite.y, range.origin.z};
    range.wVertex[1] = {range.opposite.x, range.opposite.y, range.origin.z};
    range.wVertex[2] = {range.opposite.x, range.origin.y, range.origin.z};
    range.wVertex[3] = {range.origin.x, range.origin.y, range.origin.z};
    range.wVertex[4] = {range.origin.x, range.opposite.y, range.opposite.z};
    range.wVertex[5] = {range.opposite.x, range.opposite.y, range.opposite.z};
    range.wVertex[6] = {range.opposite.x, range.origin.y, range.opposite.z};
    range.wVertex[7] = {range.origin.x, range.origin.y, range.opposite.z};

    loadStatus = ENDED_PROCESS;

    #ifdef DEBUG_FILE_

    char buffer[256];

    OutputDebugStringA("\n");
    OutputDebugStringA("v world\n");
    for (int i = 0; i < v.world.size(); ++i)
    {
        sprintf_s(buffer, "%f", v.world[i].x);
        OutputDebugStringA(buffer);
        OutputDebugStringA(" ");

        sprintf_s(buffer, "%f", v.world[i].y);
        OutputDebugStringA(buffer);
        OutputDebugStringA(" ");

        sprintf_s(buffer, "%f", v.world[i].z);
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }

    OutputDebugStringA("\n");
    OutputDebugStringA("range origin\n");
    sprintf_s(buffer, "%f", range.origin.x);
    OutputDebugStringA(buffer);
    OutputDebugStringA(" ");

    sprintf_s(buffer, "%f", range.origin.y);
    OutputDebugStringA(buffer);
    OutputDebugStringA(" ");

    sprintf_s(buffer, "%f", range.origin.z);
    OutputDebugStringA(buffer);
    OutputDebugStringA("\n");

    OutputDebugStringA("\n");
    OutputDebugStringA("range opposite\n");
    sprintf_s(buffer, "%f", range.opposite.x);
    OutputDebugStringA(buffer);
    OutputDebugStringA(" ");

    sprintf_s(buffer, "%f", range.opposite.y);
    OutputDebugStringA(buffer);
    OutputDebugStringA(" ");

    sprintf_s(buffer, "%f", range.opposite.z);
    OutputDebugStringA(buffer);
    OutputDebugStringA("\n");
    

    OutputDebugStringA("\n");
    OutputDebugStringA("v uv\n");
    for (int i = 0; i < v.uv.size(); ++i)
    {
        sprintf_s(buffer, "%f", v.uv[i].x);
        OutputDebugStringA(buffer);
        OutputDebugStringA(" ");

        sprintf_s(buffer, "%f", v.uv[i].y);
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }

    OutputDebugStringA("\n");
    OutputDebugStringA("v normal\n");
    for (int i = 0; i < v.normal.size(); ++i)
    {
        sprintf_s(buffer, "%f", v.normal[i].x);
        OutputDebugStringA(buffer);
        OutputDebugStringA(" ");

        sprintf_s(buffer, "%f", v.normal[i].y);
        OutputDebugStringA(buffer);
        OutputDebugStringA(" ");

        sprintf_s(buffer, "%f", v.normal[i].z);
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }

    OutputDebugStringA("\n");
    OutputDebugStringA("poly v\n");
    for (int i = 0; i < poly.v.size(); ++i)
    {
        sprintf_s(buffer, "%d", poly.v[i].num1);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.v[i].num2);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.v[i].num3);
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }

    OutputDebugStringA("\n");
    OutputDebugStringA("poly uv\n");
    for (int i = 0; i < poly.uv.size(); ++i)
    {
        sprintf_s(buffer, "%d", poly.uv[i].num1);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.uv[i].num2);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.uv[i].num3);
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }

    OutputDebugStringA("poly normal\n");
    OutputDebugStringA("\n");
    for (int i = 0; i < poly.normal.size(); ++i)
    {
        sprintf_s(buffer, "%d", poly.normal[i].num1);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.normal[i].num2);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.normal[i].num3);
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }

    #endif
    

    file.close();
    loadStatus = LOAD_SUCCESS;
    return 0;
}

int MTL_FILE::loadData(std::string inputFileName)
{
    std::string filePath ("../loadfiles/");
    fileName = inputFileName;

    filePath.append("mtl");
    filePath.append("/");
    filePath.append(fileName);

    std::ifstream file(filePath);

    if (file.fail())
    {
        #ifdef DEBUG_FILE_

        OutputDebugStringA("mtl file open failed\n");

        #endif
        loadStatus = LOAD_FAILURE;
        return 1;
    }

    loadStatus = STANDBY_LOAD;

    std::string tag;
    std::string line;
    std::string name;
    std::size_t punc1;
    std::size_t punc2;
    std::size_t punc3;
    VECTOR3D num3d;

    loadStatus = LOADING;

    while (std::getline(file, line)) {
        punc1 = line.find(" ");
        tag = line.substr(0, punc1);

        // Branching by TAG
        if (tag == "Ka")
        {
            // Ambient light

            // Save the first number
            punc2 = line.find(" ", tag.size() + 2);
            num3d.x = std::stod(line.substr(tag.size() + 1, punc2 - (tag.size() + 1)));

            // Save the second number
            punc3 = line.find(" ", punc2 + 1);
            num3d.y = std::stod(line.substr(punc2 + 1, punc3 - (punc2 + 1)));
            
            // Save the third number
            num3d.z = std::stod(line.substr(punc3 + 1, line.size() - (punc3 + 1)));

            ka = num3d;
        }
        else if (tag == "Kd")
        {
            // Diffuse light

            // Save the first number
            punc2 = line.find(" ", tag.size() + 2);
            num3d.x = std::stod(line.substr(tag.size() + 1, punc2 - (tag.size() + 1)));

            // Save the second number
            punc3 = line.find(" ", punc2 + 1);
            num3d.y = std::stod(line.substr(punc2 + 1, punc3 - (punc2 + 1)));
            
            // Save the third number
            num3d.z = std::stod(line.substr(punc3 + 1, line.size() - (punc3 + 1)));

            kd = num3d;
        }
    }

    #ifdef DEBUG_FILE_

    char buffer[256];

    OutputDebugStringA("\n");
    OutputDebugStringA("mtl ka\n");
    sprintf_s(buffer, "%f", ka.x);
    OutputDebugStringA(buffer);
    OutputDebugStringA(" ");

    sprintf_s(buffer, "%f", ka.y);
    OutputDebugStringA(buffer);
    OutputDebugStringA(" ");

    sprintf_s(buffer, "%f", ka.z);
    OutputDebugStringA(buffer);
    OutputDebugStringA("\n");

    OutputDebugStringA("\n");
    OutputDebugStringA("mtl kd\n");
    sprintf_s(buffer, "%f", kd.x);
    OutputDebugStringA(buffer);
    OutputDebugStringA(" ");

    sprintf_s(buffer, "%f", kd.y);
    OutputDebugStringA(buffer);
    OutputDebugStringA(" ");

    sprintf_s(buffer, "%f", kd.z);
    OutputDebugStringA(buffer);
    OutputDebugStringA("\n");

    #endif
    
    file.close();
    loadStatus = LOAD_SUCCESS;
    return 0;
}

// BMP files
BMP_FILE sampleBmpFile;

// // OBJ files
// OBJ_FILE tempObjFile;

// MTL files
MTL_FILE tempMtlFile;
