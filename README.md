# RemoteAnimalScan
Remote Animal Scan is a workflow that will provide tools to collect and analyze image, pointcloud and geospatial data.


# Label tool:

![](https://github.com/hobbitsyfeet/RemoteAnimalScan/raw/main/Logo_Final.png)

![image](https://user-images.githubusercontent.com/11593824/212435037-86d0c743-db3f-4783-866a-4f0ecbdad9fe.png)

# Create Dataset tool:

![Screenshot 2023-01-13 161030](https://user-images.githubusercontent.com/11593824/212435373-0faa1a1e-bce1-41b0-87eb-f52d501f69c9.png)

# Dependencies for kinect transform (https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/.gitmodules)
cd K:\Github\Azure-Kinect-Sensor-SDK-develop\extern\libjpeg-turbo\src
git clone https://github.com/libjpeg-turbo/libjpeg-turbo
cd src
make a folder called build
cd build
cmake ..
open libjpeg-turbo.sln
Right click ALL_BUILD -> Configuration Properties -> General Project Defaults -> Configuration Type _> Static library (.lib)
right click ALL BUILD and build it


Kinect to ply example from https://github.com/Microsoft/Azure-Kinect-Sensor-SDK/tree/develop/examples/transformation
(Windows)
1. Download the sdk source code from github (git clone https://github.com/microsoft/Azure-Kinect-Sensor-SDK)
2. navigate to Azure-Kinect-Sensor-SDK-develop\examples\transformation\
3. make build folder inside of Azure-Kinect-Sensor-SDK-develop\examples\transformation\ and enter it
4. cmake ..
5. open Project.sln inside of visual studios
6. Click on transformation_example under Solution Explorer (View->Solution Explorer)
7. include nuget compiled sources from Project-> Manage NuGet Packages... -> Browse -> search K4AdotNet -> Hit download arrow on right and install
8. right click transformation_example -> Properties -> C/C++ -> Additional Include Directories 
9. Add this folder (or one relatively similar given the version): Azure-Kinect-Sensor-SDK-develop\src\csharp\packages\Microsoft.Azure.Kinect.Sensor.1.4.1\build\native\include\
10. add to Additional Include Directories: K:\Github\Azure-Kinect-Sensor-SDK-develop\extern\libjpeg-turbo\src\libjpeg-turbo 
11. Navigate to transformation_example (right click) -> Properties -> Linker -> Input
12. Navigate to K:\Github\Azure-Kinect-Sensor-SDK-develop\src\csharp\packages\Microsoft.Azure.Kinect.Sensor.1.4.1\lib\native\amd64\release\
Note: Select your appropriate version
13 Azure-Kinect-Sensor-SDK-develop\extern\libjpeg-turbo\src\libjpeg-turbo\build\Debug\turbojpeg.lib
14. Remove libjpeg-turbo.lib from Linker -> Input Additional Dependencies, line 13 should make up for this.
15. because .lib is removed, you need to copy Azure-Kinect-Sensor-SDK-develop\extern\libjpeg-turbo\src\libjpeg-turbo\build\Debug to beside transformation_example.exe to work.
