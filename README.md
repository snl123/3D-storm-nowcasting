# 3D-storm-nowcasting
This is a TensorFlow implementation of 3D-ConvLSTM, a three-dimensional gridded radar echo extrapolation model for convective storm nowcasting as described in the following paper:

* [Three-Dimensional Gridded Radar Echo Extrapolation for Convective Storm Nowcasting Based on 3D-ConvLSTM Model](http://https://www.mdpi.com/2072-4292/14/17/4256), by Nengli Sun, Zeming Zhou, Qian Li, and Jinrui Jing.

If you use this method or this code in your research, please cite as:

```
@Article{rs14174256,
AUTHOR = {Sun, Nengli and Zhou, Zeming and Li, Qian and Jing, Jinrui},
TITLE = {Three-Dimensional Gridded Radar Echo Extrapolation for Convective Storm Nowcasting Based on 3D-ConvLSTM Model},
JOURNAL = {Remote Sensing},
VOLUME = {14},
YEAR = {2022},
NUMBER = {17},
ARTICLE-NUMBER = {4256},
DOI = {10.3390/rs14174256}
}
```

## Dependencies
In general, several major packages are needed\
python==3.7.2\
tensorflow-gpu==1.14.0\
tensorflow-determinism==0.3.0\
numpy==1.21.6

## References and Github Links
[1] Mustafa, M.A. A data-driven learning approach to image registration. University of Nottingham. 2016.("https://github.com/Mustafa3946/Lucas-Kanade-3D-Optical-Flow")

[2] Ayzel, G.; Heistermann, M.; Winterrath, T. Optical flow models as an open benchmark for radar-based precipitation nowcasting (rainymotion v0.1). Geoscientific Model Development 2019, 12, 1387-1402.("https://github.com/hydrogo/rainymotion")

[3] Pulkkinen, S.; Nerini, D.; Pérez Hortal, A.A.; Velasco-Forero, C.; Seed, A.; Germann, U.; Foresti, L. Pysteps: An open-source python library for probabilistic precipitation nowcasting (v1.0). Geoscientific Model Development 2019, 12, 4185-4219.("https://github.com/pySTEPS/pysteps")

[4] Wang, Y.; Long, M.; Wang, J.; Gao, Z.; Yu, P.S. Predrnn: Recurrent neural networks for predictive learning using spatiotemporal lstms. Advances in Neural Information Processing Systems, 2017; pp. 879–888.("https://github.com/Yunbo426/predrnn-pp")

[5] Shi, X.; Chen, Z.; Wang, H.; Yeung, D.Y.; Wong, W.K.; Woo, W.-c. Convolutional LSTM network: A machine learning approach for precipitation nowcasting. Advances in Neural Information Processing Systems, 2015; pp. 802–810.
