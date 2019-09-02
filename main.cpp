#include "system/system.h"
#include <memory>
int main(int argc, char ** argv)
{
    if (argc < 2) // first is name of program
    {
        I3D_LOG(i3d::error) << "Not enough input arguments: REVO configFile.yaml datasetFile.yaml";
        exit(0);
    }
    const std::string settingsFile = argv[1], datasetFile = argv[2];
    int nRuns = 0;

    while (true)
    {
        REVO revoSystem(settingsFile,datasetFile,nRuns);

        if (!revoSystem.start())
        {
            I3D_LOG(i3d::info) << "Finished all datasets!";
            return EXIT_SUCCESS;
        }
        nRuns++;
        I3D_LOG(i3d::info) << "nRuns: " << nRuns <<"\n";

    }
    return EXIT_SUCCESS;
}
