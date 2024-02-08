#include "readCIFAR10.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

bool fileExists(const std::string& path)
{
    std::ifstream f(path.c_str());
    if (!f.good())
    {
        std::cout << "File \"" << path
                  << "\" does not exists, please download it." << std::endl;
        return false;
    }
