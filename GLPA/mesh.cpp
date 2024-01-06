#include "mesh.h"


void Mesh::load(std::string fileName, std::string folderPass){
    std::string filePath = folderPass + "/" + fileName;

    std::ifstream file(filePath);

    if (file.fail())
    {
        std::runtime_error(ERROR_MESH_LOAD_FILE);
        return;
    }

    // Initialize data
    range.status = false;

    v.world.resize(0);
    v.uv.resize(0);
    v.normal.resize(0);

    poly.vId.resize(0);
    poly.uvId.resize(0);
    poly.normalId.resize(0);


    std::string tag;
    std::string line;
    std::string name;
    std::size_t punc1;
    std::size_t punc2;
    std::size_t punc3;
    std::size_t punc4;
    Vec3d num3d;
    Vec2d num2d;
    NumComb3 numComb3V;
    NumComb3 numComb3UV;
    NumComb3 numComb3Normal;



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
            numComb3V.n1 = std::stoi(line.substr(tag.size() + 1, punc2 - (tag.size() + 1))) - 1;

            // Save the second number
            punc3 = line.find("/", punc2 + 1);
            numComb3UV.n1 = std::stoi(line.substr(punc2 + 1, punc3 - (punc2 + 1))) - 1;

            // Save the third number
            punc4 = line.find(" ", punc3 + 1);
            numComb3Normal.n1 = std::stoi(line.substr(punc3 + 1, punc4 - (punc3 + 1))) - 1;

            // Save uv numbers
            // Save the first number
            punc2 = line.find("/", punc4 + 1);
            numComb3V.n2 = std::stoi(line.substr(punc4 + 1, punc2 - (punc4 + 1))) - 1;

            // Save the second number
            punc3 = line.find("/", punc2 + 1);
            numComb3UV.n2 = std::stoi(line.substr(punc2 + 1, punc3 - (punc2 + 1))) - 1;

            // Save the third number
            punc4 = line.find(" ", punc3 + 1);
            numComb3Normal.n2 = std::stoi(line.substr(punc3 + 1, punc4 - (punc3 + 1))) - 1;

            // Save normal numbers
            // Save the first number
            punc2 = line.find("/", punc4 + 1);
            numComb3V.n3 = std::stoi(line.substr(punc4 + 1, punc2 - (punc4 + 1))) - 1;

            // Save the second number
            punc3 = line.find("/", punc2 + 1);
            numComb3UV.n3 = std::stoi(line.substr(punc2 + 1, punc3 - (punc2 + 1))) - 1;

            // Save the third number
            numComb3Normal.n3 = std::stoi(line.substr(punc3 + 1, line.size() - (punc3 + 1))) - 1;

            poly.vId.push_back(numComb3V);
            poly.uvId.push_back(numComb3UV);
            poly.normalId.push_back(numComb3Normal);

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
        sprintf_s(buffer, "%d", poly.v[i].n1);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.v[i].n2);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.v[i].n3);
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }

    OutputDebugStringA("\n");
    OutputDebugStringA("poly uv\n");
    for (int i = 0; i < poly.uv.size(); ++i)
    {
        sprintf_s(buffer, "%d", poly.uv[i].n1);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.uv[i].n2);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.uv[i].n3);
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }

    OutputDebugStringA("poly normal\n");
    OutputDebugStringA("\n");
    for (int i = 0; i < poly.normal.size(); ++i)
    {
        sprintf_s(buffer, "%d", poly.normal[i].n1);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.normal[i].n2);
        OutputDebugStringA(buffer);
        OutputDebugStringA("/");

        sprintf_s(buffer, "%d", poly.normal[i].n3);
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }

    #endif
    

    file.close();
}
