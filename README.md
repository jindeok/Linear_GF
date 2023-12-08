# Graph Filter (linear)
* Extremely faster implementation of **linear low-pass graph filter** part in GF-CF [Shen et al. CIKM'21]
* Specifically, we use sparse matrices & GPU acceleration for the inference
* You can find the original implementation of full GF-CF method in: https://github.com/yshenaw/GF_CF


# Dependancy
* numpy
* Pytorch
* Scipy 

# Notes
* You can simply test & run cells in 'Graph Filtering (Light).ipynb'
* Our code provides 'MovieLens-1M', 'Yelp', 'Gowalla', and 'Amazon-Book' datasets
* One can possibly customize the code to implement batch inference-fashion based on your preference

