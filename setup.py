from setuptools import setup, find_packages

# read the contents of your README file
from os import path
import io

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="visual_AI_HPE",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "EasyDict",
        "torch",
        "opencv-python",
        "Cython",
        "scipy",
        "json_tricks",
        "scikit-image",
        "torchvision",
        "matplotlib",
        "timm",
        "einops",
        "wandb",
        "pycocotools"
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="Human Pose Estimation with uncertainty Library",
    author="Daejong Jin, Juhan Park",
    url="",
    download_url="p",
    author_email="",
    version="",
    long_description=long_description,
    long_description_content_type="text/markdown",
)

# python setup.py sdist bdist_wheel
# twine upload --skip-existing dist/*
