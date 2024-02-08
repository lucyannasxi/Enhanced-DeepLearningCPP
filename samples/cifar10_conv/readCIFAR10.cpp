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

    return true;
}

void parse(const std::string& path, std::vector<std::vector<float>>& xs,
           std::vector<std::vector<float>>& ys)
{
    std::ifstream file(path);
    for (int n = 0; n < 10000; ++n)
    {
        std::vector<float> x(3072);
        std::vector<float> y(10, 0.);
        char byte;

        file.read(&byte, sizeof(char));
        y[int(byte)] = 1.;

        for (int c = 0; c < 3; ++c)
        {
            for (int i = 0; i < 1024; ++i)
            {
                file.read(&byte, sizeof(char));
                x[3 * i + c] = float(byte) / 127. - 1.;
            }
        }

        ys.push_back(y);
        xs.push_back(x);
    }
}

Cifar10Dataset::Cifar10Dataset(const std::vector<std::string>& paths,
                               int batchSize)
    : mBatchSize(batchSize), mPos(0)
{
    for (const auto& path : paths) assert(fileExists(path));

    for (const auto& path : paths) parse(path, mX, mY);
    mIndexes = std::vector<int>(mX.size());
    std::iota(mIndexes.begin(), mIndexes.end(), 0);
    std::shuffle(mIndexes.begin(), 