# Image Similarity

This project looks at creating a general purpose class (Img2Vec) for assessing image similarity. 
The approach taken looks to embed images by extracting the final layer feature embeddings from 
pretrained neural networks, before comparing similarity between two images based on Cosine distance.

The class has been written to accomodate some standard neural network architectures, supported by in
Pytorch's torchvision library. At time of publishing these include: resnet50, vgg19, and efficientnet_b0

The main.ipynb contains a brief demonstration of how to utilise the Img2Vec class functionality. 
The class itself can be found in the image_similarity.py module within the "ImgSim" package directory.

## Initial setup
The first thing you are going to want to do is to clone this repository.

```
$ https://github.com/totogot/ImageSimilarity.git
```

Next, set up a virtual environment for installing all package requirements into.

```
$ cd C:\Users\jdoe\Documents\PersonalProjects\image_similarity
$ python -m venv venv
```

Then from within the terminal command line within your IDE (making sure you are in the project folder), you can install all the dependencies for the project, by simply activating the venv and leveraging the setuptools package and the setup.cfg file created in the project repo. 

```
$ .\venv\Scripts\activate
$ pip install --upgrade pip
$ pip install .
```

This last command will install all dependencies outlined in the setup.cfg file. ipykernel has been included to enable the main.ipynb to be run also and for relevant visualisations to be outputted also.

Note: for IDEs where the CLI uses PowerShell by default (e.g. VS Code), in order to run Activate.ps1 you may find that you first need to update your settings such that Command Prompt is the default terminal shell - see here: https://support.enthought.com/hc/en-us/articles/360058403072-Windows-error-activate-ps1-cannot-be-loaded-because-running-scripts-is-disabled-UnauthorizedAccess-


## Data
For the purpose of demonstrating functionality I have included several test images (in this case flowers) 
*DISCLAIMER: The images were sourced from the web, and I do not claim to hold any rights to copyrights or other forms of IP.

The folder structure is as follows:
./data/flower_images/main_imgs = main directory of images that will be embedded to form the core dataset in which to search
./data/flower_images/target_imgs = contains target images for which we will look to find most similar images in main dataset


## Repository formatting
It is worth noting that pre-commit functionality has been included in this repository.
"Flake8" and "black" functionality has been included - specified by .pre-commit-config.yaml file.

If making future edits to a cloned repo, and you wish to access this functionality, you will need to execute the following,
after install all requirements, and prior to making future commits:

```
$ pre-commit install
```