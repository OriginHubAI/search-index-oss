#include "Utils.h"
#include <fstream>
#include <sys/resource.h>

namespace Search
{

std::string getRSSUsage()
{
    std::string line;

    // Create the file path for the process status
    std::string path = "/proc/self/status";

    // Open the file for reading
    std::ifstream file(path);
    if (file.is_open())
    {
        // Search for the line containing the RSS
        while (std::getline(file, line))
        {
            if (line.substr(0, 5) == "VmRSS")
            {
                return line;
            }
        }
        file.close();
    }
    else
    {
        SI_LOG_ERROR("Error opening file.");
        return "";
    }
    return "";
}

void printMemoryUsage(const std::string & header)
{
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    SI_LOG_INFO(
        "{} Current memory usage: {}, peak memory usage: {} MB",
        header,
        getRSSUsage(),
        usage.ru_maxrss / 1024);
}

}