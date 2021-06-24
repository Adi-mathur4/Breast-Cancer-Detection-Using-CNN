# Breast-Cancer-Detection-Using-CNN
A CNN model trained using BreakHis dataset which can identify  and differentiate between malignant and benign cells with an accuracy of 95.33%.

#BreakHis Dataset
The Breast Cancer Histopathological Image Classification (BreakHis) is  composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X).
To date, it contains 2,480  benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format).

First of all the dataset is divided into two parts i.e. training set and testing set. (Reffer mkfold.py)

For all the images, the contrast has been increased using the CLAHE in order to get more data from the each and every image that can result in increasing the accuracy of the model.

The model took almost 27 hrs. to get trained and the provided dataset and after training, the model was able to achieve an accuracy of 95.33%.
