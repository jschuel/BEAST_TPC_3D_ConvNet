# BEAST_TPC_3D_ConvNet
Example repository containing data processing and analysis tools for vector directional assignment of TPC recoil events using 3D convolutional neural networks. We include sample data, sample trained models, a file for the network architecture, a custom PyTorch DataLoader class, and training/evaluation scripts.

The trained models were trained using ~500,000 images with 10-fold cross validation. The sample we include here only has 10,000 images. which is still a large enough sample to train your own model if you would like, however you won't get as good of performance training using the 10,000 images as you would with our pretrained models (pretrained model weight files can be found in `ConvNet/models/fold*.pth`).

### Recommended usage order after cloning:
1. Extract tensors from tarball using `tar -xvf tensors.tar.gz` tensors
2. Run **`scripts/evaluate_cv.py`** **(Note: pyarrow is required to read and write feather files through pandas and can be installed using either pip or anaconda)**
3. Analyze the results of the output file. An example analysis is shown in **`Analysis/analyze_output.ipynb`**
4. **Optional**: You can also train your own models using the **`train_cv.py`** in the `**`scripts`** directory.

### Alternatively you can generate tensors yourself
1. Create an empty folder in the parent directory called `tensors`
2. Open up a jupyter notebook session and run through each cell of **`data_processing/Labeling_and_tensors.ipynb`**
3. After step (2) you will have generated tensors for all 10,000 events
4. Run **`scripts/evaluate_cv.py`**
5. Analyze the results of the output file. An example analysis is shown in **`Analysis/analyze_output.ipynb`**

#### Image of a similar, yet different network architecture used for directional classification in the BEAST TPCs:
![plot](./misc_images/architecture_example.png)
