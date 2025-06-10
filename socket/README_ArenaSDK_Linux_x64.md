README for Arena SDK for Linux

The Arena SDK documentation includes a section called "Initial Configuration in Linux". Please review this section before proceeding.


Arena SDK requires the following dependencies:

- g++ 5 or higher
- make
- libibverbs1
- librdmacm1
- libx264-dev and libx265-dev if using the Save API

Installing Arena SDK for Linux:

1. Extract the tarball to your desired location:

    $ tar -xvzf ArenaSDK_Linux.tar.gz
    
    where ArenaSDK_Linux.tar.gz is the tarball name.

2. Run the Arena_SDK.conf file

   WARNING: Pass -cti argument to set GENICAM_GENTL64_PATH environment variable.
            Reboot PC before run .cti.

    $ cd /path/to/ArenaSDK_Linux
    $ sudo sh Arena_SDK_Linux_x64.conf
    
    This will make the Arena SDK shared library files accessible by the run-time linker (ld.so or ld-linux.so).


Examples:
  
A precompiled version of the examples, including IpConfigUtility, are located in ArenaSDK_Linux/precompiledExamples.
    
- C++ examples are located in ArenaSDK_Linux/Examples/Arena. C examples are located in ArenaSDK_Linux/Examples/ArenaC. The Arena or ArenaC folders contain a Makefile that can be used to compile each example.

- Upon successful compilation, the compiled binaries will be placed into ArenaSDK_Linux/OutputDirectory/Linux/x64Release.


RDMA Support:
#############

Warning: RDMA support is available on specific models of Lucid cameras. Please check the documentation for your camera model. 
For cameras that do support RDMA streaming control, control of this feature is available using the camera's TransportStreamProtocol node.

A Broadcom driver package is included with the installer as a separate tarball: Lucid_Broadcom_Driver_Package_v*.tar.gz.
It contains the recommended Broadcom ethernet and RoCEv2 drivers for use with the Arena SDK. 

To install the package:

1. 1. Extract the tarball to your desired location:

    $ tar -xvzf Lucid_Broadcom_Driver_Package_v*.tar.gz

2. Run the install_broadcom_driver.sh script

    $ cd /path/to/Lucid_Broadcom_Driver_Package
    $ sudo ./install_broadcom_driver.sh
    

This script will install the included Broadcom Ethernet and RoCE drivers.
After the driver installation the MTU for each Broadcom interface will
be set to 9000 bytes for the current session.

Note that the MTU change made by this script is not persistant. To have
the MTU change be persistent, update the appropriate configuration file for your system.