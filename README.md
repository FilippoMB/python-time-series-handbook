# Time Series Analysis with Python

<img src="https://raw.githubusercontent.com/FilippoMB/python-time-series-handbook/main/logo.png" style="width: 5cm; display: block; margin: auto;">

<br>
<div align="center">
📚 <a href="https://filippomb.github.io/python-time-series-handbook">Read it as a book</a>
</div>
<br>

This is the collection of notebooks for the course *Time Series Analysis with Python*.
For more information and for reading the content of this repository, please refer to the [book](https://filippomb.github.io/python-time-series-handbook) version.

## 📑 Content

1. **Introduction to time series analysis**
   - Definition of time series data
   - Main applications of time series analysis
   - Statistical vs dynamical models perspective
   - Components of a time series
   - Additive vs multiplicative models
   - Time series decomposition
  
> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/01/introduction_to_time_series.ipynb)

2. **Stationarity in time series**
   - Stationarity in time series
   - Weak vs strong stationarity
   - Autocorrelation and autocovariance
   - Common stationary and nonstationary time series
   - How to identify stationarity
   - Transformations to achieve stationarity

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/02/stationarity.ipynb) 

3. **Smoothing**
   - Smoothing in time series data
   - The mean squared error
   - Simple average, moving average, and weighted moving average
   - Single, double, and triple exponential smoothing

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/03/smoothing.ipynb) 

4. **AR-MA**
   - The autocorrelation function
   - The partial autocorrelation function
   - The Auto-Regressive model
   - The Moving-Average model

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/04/ar-ma.ipynb)

5. **ARMA, ARIMA, SARIMA**
   - Autoregressive Moving Average (ARMA) models
   - Autoregressive Integrated Moving Average (ARIMA) models
   - SARIMA models (ARIMA model for data with seasonality)
   - Automatic model selection with AutoARIMA
   - Model selection with exploratory data analysis

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/05/arma_arima_sarima.ipynb)

6. **Unit root test and Hurst exponent**
   - Unit root test
   - Mean Reversion
   - Hurst Exponent
   - Geometric Brownian Motion
   - Applications in quantitative finance

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/06/unit-root-hurst.ipynb)

7. **Kalman filter**
   - Introduction to Kalman Filter
   - Model components and assumptions
   - The Kalman Filter algorithm
   - Application to static and dynamic one-dimensional data
   - Application to higher-dimensional data

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/07/kalman-filter.ipynb) 

8. **Signal transforms and filters**
   - Introduction to Fourier Transform and Discrete Fourier Transform, and FFT
   - Fourier Transform of common signals
   - Properties of the Fourier Transform
   - Signal filtering with low-pass, high-pass, band-pass, and bass-stop filters
   - Application of Fourier Transform for time series forecasting

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/08/signal-transforms-filters.ipynb)

9. **Prophet**
   - Introduction to Prophet for time series forecasting
   - Advanced modelling of trend, seasonality, and holidays
   - The Prophet library in Python

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/09/prophet.ipynb)

10. **Neural networks and Reservoir Computing**
    - Windowed approaches and Neural Networks for time series forecasting
    - Forecasting with the Multi Layer Perceptron
    - Recurrent Neural Networks: advantages and challenges
    - Reservoir Computing and the Echo State Network
    - Dimensionality reduction with Principal Component Analysis
    - Forecasting electricity consumption with Multi Layer Perceptron and Echo State Network

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/10/nn-reservoir-computing.ipynb)

11. **Non-linear time series analysis**
    - Dynamical systems and nonlinear dynamics
    - Bifurcation diagrams
    - Chaotic systems
    - High-dimensional continuous-time systems
    - Fractal dimensions
    - Phase space reconstruction and Taken's embedding theorem
    - Forecasting nonlinear time series

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/11/nonlinear-ts.ipynb)

12. **Time series classification and clustering**
    - Multivariate time series
    - Time series similarity
    - Dynamic Time Warping
    - Time series kernels
    - Time series embedding
    - Classification of time series
    - Clustering of time series
    - Visualization with kernel PCA

> [notebook](https://nbviewer.org/github/FilippoMB/python-time-series-handbook/blob/main/notebooks/12/classification-clustering.ipynb)

## 🚀 Getting started with coding

To run the notebooks the recommended steps are the following:

1. Download and install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

2. Download the [env.yml](env.yml) file.

3. Open the shell and navigate to the location with the yml file you just downloaded.
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
