from distutils.core import setup

setup(
    name="texsymdetect",
    version="0.0.3",
    packages=[
      "texsymdetect.client",
    ],
    license="Apache License 2.0",
    long_description=open("README.md").read(),
    url="https://github.com/andrewhead/texsymdetect",
    install_requires=[
        "requests>=2.0.0,<3.0.0"
    ],
)
