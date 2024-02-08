#include "readMNIST.h"

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

    return true;
}

unsigned rev(unsigned n)
{
    unsigned char c1 = n & 255;
    unsigned char c2 = (n >> 8) & 255;
    unsigned char c3 = (n >> 16) & 255;
    unsigned char c4 = (n >> 24) & 255;

    return (unsigned(c1) << 24) + (unsigned(c2) << 16) + (unsigned(c3) << 8) +
           c4;
}

void parseImages(const std::string& path, std::vector<std::vector<float>>& db)
{
    std::ifstream file(path);
    if (file.is_open())
    {
        unsigned mn, N, R, C;
        file.read(reinterpret_cast<char*>(&mn), sizeof(mn));
        mn = rev(mn);
        file.read(reinterpret_cast<char*>(&N), sizeof(N));
        N = rev(N);
