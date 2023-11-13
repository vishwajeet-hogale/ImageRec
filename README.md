# GetYourPics Tool
## Introduction<a name="introduction"></a>
Story behind this project:
At significant events like meetings, outings, or functions, people capture numerous photos that are often uploaded to a single Google Drive. Retrieving individual photos from such collective storage can be time-consuming, especially when someone wants to download all images featuring them. The Event Photo Manager addresses this issue, providing a solution for users to upload their photos and efficiently download all pictures in which they are present.

## Features<a name="features"></a>

* User Photo Upload: Enables users to upload their photos to the system.
* Face Recognition: Utilizes ResNet-152 for image recognition to identify individuals in photos.
* Face Extraction: Uses OpenCV haarcascade to extract faces of multiple people from an image.
* Dictionary Mapping: Creates a dictionary mapping each person to the images they are present in.
* Personalized Photo Download: Allows users to download a zip file containing all pictures in which they are present.

## Dependencies<a name="dependencies"></a>
* Python (version 3.6 or higher)
* OpenCV
* ResNet-152 model (pre-trained)
* Other Python libraries (requirements specified in requirements.txt) 

## Algorithm Overview<a name="algorithm-overview"></a>
### Train
1. Use the input images of a user to train the ResNet - 152 model 
2. Also apply data augmentation techniques to improve the dataset
3. Store the model
### Model execution 
1. Loop through the directory in which all pictures from the event are present
2. Use OpenCV to extract faces from group images
3. Use the model to predict the person
4. Create dictionary mappings
5. Save the mappings
6. Use the mappings to download the images for a particular individual


## Contributing<a name="contributing"></a>
Contributions are welcome! Please follow the guidelines outlined in CONTRIBUTING.md when contributing to this project.

## License<a name="license"></a>
This project is licensed under the MIT License - see the LICENSE file for details.
