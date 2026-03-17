# Time Series Analysis with Python

<div align="center">
   <img src="https://raw.githubusercontent.com/FilippoMB/python-time-series-handbook/main/logo.png" style="width: 5cm; display: block; margin: auto;">
</div>

<br>
<div align="center">
📚 <a href="https://filippomb.github.io/python-time-series-handbook">Read it as a book</a>
</div>
<br>

[![PyPI Downloads](https://static.pepy.tech/badge/tsa-course)](https://pepy.tech/projects/tsa-course)

This is the collection of notebooks for the course *Time Series Analysis with Python*.
You can view and execute the notebooks by clicking on the buttons below.

## 📑 Content

1. **Introduction to time series analysis** 
   - Definition of time series data
   - Main applications of time series analysis
   - Statistical vs dynamical models perspective
   - Components of a time series
   - Additive vs multiplicative models
   - Time series decomposition techniques

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/01/introduction_to_time_series.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/01/introduction_to_time_series.ipynb)

<br>

2. **Stationarity in time series**
   - Stationarity in time series
   - Weak vs strong stationarity
   - Autocorrelation and autocovariance
   - Common stationary and nonstationary time series
   - How to identify stationarity
   - Transformations to achieve stationarity

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/02/stationarity.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/02/stationarity.ipynb)

<br>

3. **Smoothing**
   - Smoothing in time series data
   - The mean squared error
   - Simple average, moving average, and weighted moving average
   - Single, double, and triple exponential smoothing

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/03/smoothing.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/03/smoothing.ipynb)

<br>

4. **AR-MA**
   - The autocorrelation function
   - The partial autocorrelation function
   - The Auto-Regressive model
   - The Moving-Average model
   - Reverting stationarity transformations in forecasting

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/04/ar-ma.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/04/ar-ma.ipynb)

<br>

5. **ARMA, ARIMA, SARIMA**
   - Autoregressive Moving Average (ARMA) models
   - Autoregressive Integrated Moving Average (ARIMA) models
   - SARIMA models (ARIMA model for data with seasonality)
   - Automatic model selection with AutoARIMA
   - Model selection with exploratory data analysis

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/05/arma_arima_sarima.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/05/arma_arima_sarima.ipynb)

<br>

6. **Unit root test and Hurst exponent**
   - Unit root test
   - Mean Reversion
   - The Hurst exponent
   - Geometric Brownian Motion
   - Applications in quantitative finance

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/06/unit-root-hurst.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/06/unit-root-hurst.ipynb)

<br>

7. **Kalman filter**
   - Introduction to Kalman Filter
   - Model components and assumptions
   - The Kalman Filter algorithm
   - Application to static and dynamic one-dimensional data
   - Application to higher-dimensional data

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/07/kalman-filter.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/07/kalman-filter.ipynb)

<br>

8. **Signal transforms and filters**
   - Introduction to Fourier Transform, Discrete Fourier Transform, and FFT
   - Fourier Transform of common signals
   - Properties of the Fourier Transform
   - Signal filtering with low-pass, high-pass, band-pass, and bass-stop filters
   - Application of Fourier Transform to time series forecasting

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/08/signal-transforms-filters.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/08/signal-transforms-filters.ipynb)

<br>

9. **Prophet**
   - Introduction to Prophet for time series forecasting
   - Advanced modeling of trend, seasonality, and holidays components
   - The Prophet library in Python

   [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/09/prophet.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/09/prophet.ipynb)

<br>

10. **Neural networks and Reservoir Computing**
    - Windowed approaches and Neural Networks for time series forecasting
    - Forecasting with a Multi-Layer Perceptron
    - Recurrent Neural Networks: advantages and challenges
    - Reservoir Computing and the Echo State Network
    - Dimensionality reduction with Principal Component Analysis
    - Forecasting electricity consumption with Multi-Layer Perceptron and Echo State Network

    [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/10/nn-reservoir-computing.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/10/nn-reservoir-computing.ipynb)

<br>

11. **Non-linear time series analysis**
    - Dynamical systems and nonlinear dynamics
    - Bifurcation diagrams
    - Chaotic systems
    - High-dimensional continuous-time systems
    - Fractal dimensions
    - Phase space reconstruction and Taken's embedding theorem
    - Forecasting time series from nonlinear systems

    [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/11/nonlinear-ts.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/11/nonlinear-ts.ipynb)

<br>

12. **Time series classification and clustering**
    - Multivariate time series
    - Time series similarities and dissimilarities
    - Dynamic Time Warping
    - Time series kernels
    - Embedding time series into vectors
    - Classification of time series
    - Clustering of time series
    - Visualize time series with kernel PCA

    [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/12/classification-clustering.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/12/classification-clustering.ipynb)

<br>

## 💻 How to code locally

To run the notebooks locally the recommended steps are the following:

1. Download and install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

2. Download the [env.yml](https://github.com/FilippoMB/python-time-series-handbook/blob/main/env.yml) file.

3. Open the shell and navigate to the location with the `yml` file you just downloaded.
    - If you are on Windows, open the Miniconda shell.

4. Install the environment with 
   ```{bash}
   > conda env create -f env.yml
   ```

5. Activate your environment: 
   ```{bash}
   > conda activate pytsa
   ```

6. Go to the folder with the notebooks

7. Launch Jupyter Lab with the command 
   ```{bash}
   > jupyter lab
   ```

## 🎥 Notebook format and slides

The notebooks are structured as a sequence of slides to be presented using [RISE](https://rise.readthedocs.io/en/latest/).
If you open a notebook you will see the following structure:

<img src="https://raw.githubusercontent.com/FilippoMB/python-time-series-handbook/main/notebooks/00/media/slides_nb.png" style="width: 50%" align="center">

The top-right button indicates the type of slide, which is stored in the metadata of the cell. To enable the visualization of the slide type you must first install RISE and then on the top menu select `View -> Cell Toolbar -> Slieshow`. Also, to split the cells like in the example, you must enable `Split Cells Notebook` from the [nbextensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/index.html).

By pressing the `Enter\Exit RISE Slideshow` button at the top you can enter the slideshow presentation.

<img src="https://raw.githubusercontent.com/FilippoMB/python-time-series-handbook/main/notebooks/00/media/slides_rise.png" style="width: 40%" align="center">
<img src="https://raw.githubusercontent.com/FilippoMB/python-time-series-handbook/main/notebooks/00/media/slides_rise2.png" style="width: 40%" align="center">
<img src="https://raw.githubusercontent.com/FilippoMB/python-time-series-handbook/main/notebooks/00/media/slides_rise3.png" style="width: 40%" align="center">

See the [RISE documentation](https://rise.readthedocs.io/en/latest/) for more info.

## 📝 Citation

If you are using this material in your courses or in your research, please consider citing it as follows:

````bibtex
@misc{bianchi2024tsbook,
  author       = {Filippo Maria Bianchi},
  title        = {Time Series Analysis with Python},
  year         = {2024},
  howpublished = {Online},
  url          = {https://github.com/FilippoMB/python-time-series-handbook}
}
````
