# PCA Coffee Spectra Reconstruction

This project applies dimensionality reduction using Principal Component Analysis (PCA) to reconstruct reflectance spectra of two types of coffee beans (**Arabica** and **Robusta**) and distinguish between them as well as possible.

## Installation

The notebook and script requires Python 3 and the following dependencies:

- Matplotlib (plotting results)
- NumPy (manipulating arrays and apply mathematical operations)
- Pandas (store CSV data as a dataframe)
- Scikit-learn (implement out-of-the-box PCA model)

```bash
pip install matplotlib
pip install numpy
pip install pandas
pip install scikit-learn
```

## Data 

A Fourier-Transform Infrared Spectrometer (FTIR) was used to determined the reflectance spectra of various samples of coffee beans. 

FTIR spectrometers are versatile tools used across industries for analyzing molecular compositions based on infrared absorption. This is done by by measuring the reflectance at different wavelengths, and generating a spectra which is typically unique for a particular sample. They are employed in organic synthesis, polymer science, petrochemical engineering, pharmaceuticals, food analysis, environmental monitoring, forensics, and materials science. 

The dataset used in this project contains 56 samples of coffee beans, with the first 29 spectra for Arabica and the remaining for Robusta. The wavelengths are given in the first row of the CSV file.

## Usage

'Reconstruction.ipynb' is the main file for this project, with 'pca_functions.py' containing helper functions.

To run the project, follow these steps:
- Clone the repository or download it as a zip folder and extract all files.
- Ensure you have installed the required dependencies.
- Run the Reconstruction.ipynb notebook and ensure the relative path to the CSV file is correct.

## Algorithm Overview
Principal Component Analysis (PCA) is a dimensionality reduction technique widely used in data science and machine learning. Its main objective is to identify patterns in data and represent it in a more concise form, while retaining as much information as possible.

The algorithm operates as follows:

**Data Preprocessing**
- PCA operates on numerical data, so data must be appropriately preprocessed, including handling missing values, scaling, and normalization. 
- In our case, we centered the data by subtracting column means the original dataset.

**Calculate the Covariance Matrix**
- Compute the covariance matrix of the centered data. The covariance matrix provides insights into the relationships between different features in the dataset.

**Compute Eigenvectors and Eigenvalues**
- Perform eigendecomposition on the covariance matrix to obtain its eigenvectors and corresponding eigenvalues. 
- These eigenvectors represent the directions of maximum variance in the data, and the eigenvalues indicate the magnitude of variance along these directions.
- In our case, we used the Power and Deflate methods (functions can be found in 'pca_functions.py') to numerically find eigenpairs. This involves calculating the largest eigenpair for a given matrix and iteratively removing the corresponding eigenvector from the matrix in place to find the next largest and so on.

**Select Principal Components**
- Sort the eigenvectors by their corresponding eigenvalues in descending order. These eigenvectors are called principal components. 
- Select the top k eigenvectors based on explained variance.
    
**Projection onto the Principal Components**
- Transform the original data onto the new feature subspace spanned by the selected principal components. 
- This is typically done by multiplying the original data matrix by the matrix of eigenvectors corresponding to the selected principal components.

## Benefits of PCA

**Dimensionality Reduction**: PCA reduces the number of dimensions (features) in the data while preserving most of its important information.

**Feature Extraction**: PCA helps in identifying the most informative features in the data by emphasizing the directions of maximum variance.

**Noise Reduction**: By focusing on the principal components with the highest variances, PCA can mitigate the effects of noise in the data.

**Applications**: PCA can be applied in various domains such as image processing, pattern recognition, and finance. It's commonly used as a preprocessing step before applying machine learning algorithms to high-dimensional datasets.

**Implementation**: PCA implementations are available in many machine learning libraries, such as scikit-learn (Python) and MATLAB. These libraries provide efficient and optimized implementations of PCA algorithms, making it easy to apply PCA to your datasets.

## Results
After applying PCA to the FTIR Coffee Spectra, we used **6** principal components to explain around **99.2%** of variation in the data.

## References

- Data source: Romain Briandet, E. Katherine Kemsley, and Reginald H. Wilson. “Discrimination of Arabica and Robusta in Instant Coffee by Fourier Transform Infrared Spectroscopy and Chemometrics”. In: Journal of Agriculture and Food Chemistry 44.1 (1996), pp. 170–174. doi: 10.1021/jf950305a.
