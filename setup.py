import setuptools


def main():
    pass

def getRequirements():
    install_requires = [
        "numpy>=1.16"
        "h5py>=2.10.0"
        "plyfile",
        "scipy",
        "tensorflow-gpu==1.14.0",
        "open3d>=0.11.2"
        "sklearn"

    ]


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Hobbitsyfeet", # Replace with your own username
    version="0.0.1",
    author="Justin Petluk",
    author_email="",
    description="Remote Animal Scan is a workflow that will provide tools to collect and analyze image, pointcloud and geospatial data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/ras",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",        
        "Operating System :: Unix",
        "Intended Audience :: Science/Research"
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Research and Managment",
    ],
    python_requires='>=3.7',
)

if __name__ == "__main__":
    main()