#!/bin/sh
# Shell Scripts for Build/Install Azure Kinect Sensor SDK from Source Code
## Define Version
K4A_VERSION=1.4.0
## Check CPU Architecture
ARCH=$(uname -m)
if [ $ARCH != "aarch64" ] && [ $ARCH != "x86_64" ]; then
  exit 1
fi
## Install Dependencies
sudo apt update
sudo apt install -y unzip
sudo apt install -y wget
sudo apt install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev
sudo apt install -y openssl
sudo apt install -y ninja-build
sudo apt install -y libssl-dev
sudo apt install -y libsoundio-dev
sudo apt install -y libxinerama-dev
sudo apt install -y libsdl2-dev
## Install Depth Engine
wget https://www.nuget.org/api/v2/package/Microsoft.Azure.Kinect.Sensor/$K4A_VERSION -O microsoft.azure.kinect.sensor.nupkg
mv microsoft.azure.kinect.sensor.nupkg microsoft.azure.kinect.sensor.zip
if [ $ARCH = "aarch64" ]; then
  unzip -j -o microsoft.azure.kinect.sensor.zip linux/lib/native/arm64/release/libdepthengine.so.2.0
elif [ $ARCH = "x86_64"]; then
  unzip -j -o microsoft.azure.kinect.sensor.zip linux/lib/native/x64/release/libdepthengine.so.2.0
fi
sudo chmod a+rwx /lib/$ARCH-linux-gnu
cp -f libdepthengine.so.2.0 /lib/$ARCH-linux-gnu
## Clone Azure Kinect Sensor SDK
git clone https://github.com/microsoft/Azure-Kinect-Sensor-SDK.git
cd Azure-Kinect-Sensor-SDK
git checkout -b v$K4A_VERSION refs/tags/v$K4A_VERSION
## Build and Install Azure Kinect Sensor SDK
mkdir -p build
cd build
cmake .. -GNinja
cmake --build .
sudo cmake --build . --target install
## Set Udev Rule
sudo chmod a+rwx /etc/udev/rules.d
cp -f ../scripts/99-k4a.rules /etc/udev/rules.d
## Set Emvironment Valiable for Shared Library (Temporary) 
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
#export PATH=$PATH:/usr/local/bin
## Set Emvironment Valiable for Shared Library
sudo chmod a+rwx /etc/ld.so.conf.d
echo '/usr/local/lib' > /etc/ld.so.conf.d/k4a.conf
sudo ldconfig
## Set Emvironment Valiable for Executable
echo 'PATH=$PATH:/usr/local/bin' >> ~/.bashrc
. ~/.bashrc
## Run Azure Kinect Viewer
k4aviewer
