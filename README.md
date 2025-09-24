# Multi-Label Thoracic Disease Detection from Chest X-Rays

This project focuses on the detection of multiple thoracic diseases from chest X-ray images using deep learning. The goal is to build a model that can accurately identify the presence of one or more of the following 14 pathologies: *Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax*.

## 📖 Dataset

The model is trained on the **ChestX-ray14 dataset** provided by the NIH. This dataset contains over 112,120 frontal-view X-ray images of 30,805 unique patients, each labeled with one or more of the 14 thoracic diseases.



## 🤖 Methodology

The core of this project is a **ResNet-50 model**, a powerful convolutional neural network pre-trained on the ImageNet dataset. We use **transfer learning** to adapt this model for our specific task of medical image classification.

The architecture is as follows:
1.  **Input:** A chest X-ray image.
2.  **Base Model:** A pre-trained ResNet-50, with the final classification layer removed.
3.  **Custom Head:** A new set of layers is added on top of the ResNet-50 base:
    * A Global Average Pooling 2D layer.
    * A fully connected Dense layer with 256 units and a ReLU activation function.
    * A Dropout layer for regularization.
    * A final Dense layer with 14 units (one for each disease) and a sigmoid activation function for multi-label classification.
4.  **Output:** A probability score for each of the 14 diseases.

The model is trained to minimize the **binary cross-entropy loss**, and the Adam optimizer is used.

## 📈 Results

The model's performance is evaluated using the **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)** for each of the 14 disease classes. The model achieves promising results, demonstrating its ability to effectively identify various thoracic pathologies from chest X-ray images.



## 🚀 How to Run the Code

To run the project, you will need to:
1.  Clone this repository.
2.  Download the ChestX-ray14 dataset from the official NIH website.
3.  Organize the data into the appropriate folder structure.
4.  Run the `gxsa-transfer-learning-with-resnet-50.ipynb` Jupyter notebook.

## 🛠️ Dependencies

The main dependencies for this project are:
* TensorFlow
* Keras
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## 👨‍💻 Contributors

* **Shikhar Samrat**

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
