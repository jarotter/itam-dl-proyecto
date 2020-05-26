import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="proyecto-dl-itam-jarotter", 
    version="0.1.0",
    author="Jorge Rotter",
    author_email="jorgearotter@gmail.com.com",
    description="Text to image with generative models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.1',
)