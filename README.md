# CellMembraneSegmentation_MLOps

This project focuses on implementing the U-Net architecture for biomedical image segmentation tasks using deep learning techniques. The U-Net model
is a convolutional neural network renowned for its ability to accurately segment images even when trained on limited data. Our
objective is to train, test, and visualize the U-Net model on various biomedical image datasets. We will utilize Python programming
language and TensorFlow deep learning framework to write the code.

## Project contents
The contents of this project are available in the data and source folder. Also, there are other folders (test - application - Mlflow) that are for the deployment process, each folder handles one docker image. Also, there is another folder .github/workflow which is for CI/CD and GitHub action. All the data extracted and used in this project are given in the data folder. In the source folder, you can find the codes related to the demo project, data preprocessing, model implementation, and deployment-related (last version - with AB testing) codes. This project divided into 3 different phases: *Data Preprocessing* - *Model Development* - *Deployment*

## Phase1 - Data Preprocessing
The original dataset is from [the isbi](http://brainiac2.mit.edu/isbi_challenge/) challenge, and I've downloaded it and done the pre-processing (Data augmentation and many other image-related processes). Also, I train a fine-tuned Resnet model to identify users' uploaded image class(CELL MEMBRANE or OTHER).

## Phase2 - Model development

### Classification as preprocessing

### Segmentation by U-net

## Phase3 - Deployment

### Model Deployment
#### *Deployment Strategy*
#### *Deployment Pattern*
### Automation
#### *CI/CD*
Not yet
#### *Dockerization*
#### *MLFlow*
### Monitoring
#### *Metrics*
#### *Tracking UI*

## Demo
There are two demos, one related to [local deployment](https://drive.google.com/file/d/11Pys3JW5WwitAc69QxSkiD6hlY4v1c8d/view?usp=sharing) and one related to [Darkube Deployment](https://drive.google.com/file/d/1_9JM-J7y_hdN9z06lCG4VXejnbDlp-5S/view?usp=sharing)  

## Usage
You Can use this project by this [link](https://application.darkube.app) - But first we should manually replace the dropbox_access_token in [this file](https://github.com/Amin-Saeidi/MLSD_BIOSEG/blob/master/application/DB_DropBox_Works/DB_DB_Works.py) with a new one from dropbox app console.

## Contributing
Just to let you know, pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Team
[Amin Saeidi](https://github.com/Amin-Saeidi)
## License
[MIT](https://choosealicense.com/licenses/mit/)
