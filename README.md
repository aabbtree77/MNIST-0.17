## Matuzas' Deep Network

This is the fork of an excellent work by Jonas Matuzas who has created one of the best MNIST digit classifiers existing to date (world record error rate, simple network, fast training). Here I will add some details missing in the original repo.

Firstly, lets see the resulting errors (I average only three networks to save computing time, so the error is 0.18, not 0.17 - 0.16, first label is a true class, second - best prediction, third - secondary prediction):

<table>
<tr>
<th style="text-align:center"> Test Errors</th>
</tr>
<tr>
<td>
<img src="./MNIST0.18.png"  alt="The 1st digit above each subfigure - true class label, the 2nd - prediction, followed by the second best prediction." width="100%" >
</td>
</tr>
</table>


Is the MNIST digit classification a solved problem? In the sense of the second allowed best prediction, yes, completely, no otherwise. Subjectively:

- Still recognizable: (3,5,3), (6,0,6), (7,1,7), (8,2,8), (9,7,9), (5,6,5), that would push 0.18% to 0.12%.
- Harder perhaps: (6,1,6), (6,8,6), (2,7,2), (1,7,1), pushing it to 0.08%.
- 4s vs 9s are hopeless.

Notice how these deep networks do not develop a generative mechanism of handwriting the way humans do. A human can (with some stretch) see that it is the digit five, not six in the (5,6,5) example by noticing an angle in the turn of the upper stroke/bar which is some deformation of what is supposed to approach 90 degrees in some printed font variant. The fact that the lower part is wiggled randomly from a fast hand movement and now is closer to the digit six rather than five becomes irrelevant, but how would a machine know it. 

This type of generative information comes from the writing process itself which is hard to capture with affine deformations applied on image-only data (think of the UNIPEN handwriting data sets and those touchpads where such information is available).

Some key features of Matuzas' network:

- Invoking nvidia-smi shows that without the code running my GTX760 uses 384MiB/1998MiB of its VRAM, and with this code - 1866MiB/1998MiB, so it fits into 2GB VRAM. The batch size is only 32 though. 

- A single network is trained in only about 1h20m on a desktop PC of 2015 (i7, 16GB of RAM, Ubuntu 20.04, GTX760). The number of epochs is extremely small, i.e. 20. A single prediction of the whole test set takes 13s.

- The number of adaptive layers (input-weights-activation, not counting pooling, batchnorm and such) is **15**. There are skip connections, only 3x3 convolutions applied. Interestingly, no max-pooling, average-pooling instead!

## A Note on the Best Classical (Non-Deep Network) Result

**[Rodrigo Benenson's list](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html) is not up to date and is unfair w.r.t. classics which got trully surpassed by convnets on MNIST only very recently, around 2018-2021.

To my knowledge, the best classical (non-deep learning) system for the MNIST digit recognition is to employ a simple Gaussian kriging with max-pooled log-Gabor filters of [Peter Kovesi][Peter Kovesi]. These are my own experiments so I will provide here some more details. Kovesi's filters must be with the default parameters tuned for image reconstruction, not discrimination. Attempts to find better parameters lead to a dead end. Prior to the max-pooled log-Gabor stage an input image needs to be split into x and y Sobel-filtered channels. Kriging details: Gaussian kernel interpolator whose sigma is set to the mean distance between the input patterns. **No hyperparameters**. As the kernel matrix is too big to fit into 16 GB RAM, the [tiled Cholesky decomposition][tiled Cholesky] needs to be implemented (code upon request), but this presents no problems on the machine with 64GB of RAM which I used to have in Lugano 2014!

The error rate is **0.29%** with some capacity to go down to **0.24%**. See also [this work][bknyaz], which corroborates the power of the Gabor wavelets with the achieved 0.30% error rate.

Multiple classifiers with input deformations may push the classical error down to **0.24%**, but it is very hard to actually realize this. In particular, the following combination is notable: **5x36+1x28bbwh16+1x28shear40mp56**. This is a weighted sum (5:1:1) of the three classifiers, each classifier is trained on a separate deformed set of the MNIST training digits. Each training set consists of 60K deformed patterns obtained by applying the same deformation to every single original image. The first classifier resizes the images to 36x36, the second one resizes the digit's bounding box to 16x16, followed by the centering of the bounding box within a 28x28 matrix. The third classifier sees the images which are first x-sheared with a -0.40 factor, which results in 28x39 bitmaps. Each such bitmap is then truncated by five columns from the left and six columns from the right, resulting in a 28x28 matrix.

There are simpler ways to achieve the error of **0.26%** (no shearing involved): 32+32+28bbwh26, 32+36+28bbwh26, 36+36+28bbwh16. 

I would also note that the features of [Adam Coates et al. 2010][Adam Coates et al. 2010], i.e. the triangular encoding of patch distances, reach a solid "off the shelf" error **0.35%**. However, I have tried 50K, or even 100K filters (400K dimensional vectors), different parameter settings as well, but nothing led to anything better than 0.35%. By the way, the triangular encoding can also be replaced with a more typical Gaussian kernel-based 
conversion of distances to similarities (set sigma to the mean patch distance used in the triangular encoding), it is just that the former is more efficient and works when the feature dimension is large.

For those curious about the CIFAR-10 data set, the kernel interpolator (kriging) produces the following performance values: 80.30% (4608 features), 84.64% (100K features), and 85.70% (400K features). Local patch contrast normalization is necessary, i.e. 81.52% performance without local contrast normalization (100K features). The performance value 85.70% is probably not the limit of this method, but it is too cumbersome to reach even this value.

The case with 400K features (100K patch centroids) takes roughly 10K+10Ks. (twenty kilo-seconds!) of time for feature extraction, 54Ks. for the tiled Cholesky decomposition and linear solving, and about 12Ks. for testing. So this is very time-consuming on i7 with 16GB of RAM and GTX760, but there is a lot of opportunity for parallelizations, albeit pointless in light of convnets. By the way, float32 products might further speed up the codes when calculating the kernel entries, but the single precision is definitely not enough for the products inside the tiled Cholesky decomposition as the code barfs about nonpositive definite submatrices, this problem does not appear in the double precision. 

The biggest weakness of the classical models is that averaging or maxing-out various models obtained on deformed data sets does not improve the error rate as dramatically as convnets do. I present here the exceptional cases, but in reality a simple averaging of such classifiers trained on various deformations does not improve **0.29%**. The error rate of 0.30%-0.29% should not be hard to replicate, but 0.24% is already a different case that may involve undocumented hidden factors such as Matlab's interpolation type during the shearing of images and even image dithering may have an impact. Things close to absolute zero behave strangely.

## Advice for Someone New to Machine Learning

Perfect is the enemy of good. This is the area of exponentially diminishing returns: Nothing works (improves the error) most of the time, do not base your business/study/action plan on the outliers, do not rush to improve those error rates. We might still witness another jump into the MNIST 0.10% error rate only after, say, another ten years, or this may never happen anymore.

It is incredible how GPUs and assembling of various very old ideas have changed the history, how experimental everything has become. Much of the machine learning theory before say 2010 is now obsolete and Rev. Mr. Thomas Bayes (or even E. T. Jaynes) may now rest in peace. Most of these learning theory works were starting with "let x be in Rn", forgetting to tell that this Rn is a crappy space and we will be drawing optimal decision boundaries on a trash can. So everything works in theory with nice Bayesian decision boundaries and images, and very little in practice. With convnets we are getting a good space instead of just any Rn, and in that space everything works, even overfitting.

Again, perfect is the enemy of good. The good however is also overrated and questionable. Linear algebra is cubic and demands float64. Ill-conditioned Hessians, kernel/covariance matrices. Leo Breiman's trees were universal and elegant. Inaccurate however, and these days recursion is hardly of benefit. Instead, convnets, SGD, autograd, GPUs, these are the basic pillars of machine learning now, if not the whole numerical/logic computing or even 3D graphics some day.

## References

- [anaconda]
- [anaconda-critique]
- [compute-capability-3.0]
- [Peter Kovesi]
- [backprop]
- [Adam Coates et al. 2010]
- [bknyaz]

[anaconda]: https://docs.anaconda.com/anaconda/install/linux/
[anaconda-critique]: https://www.youtube.com/watch?v=8byjq_S28PQ
[compute-capability-3.0]: https://stackoverflow.com/questions/39023581/tensorflow-cuda-compute-capability-3-0-the-minimum-required-cuda-capability-is
[Peter Kovesi]: https://www.peterkovesi.com/matlabfns/
[backprop]: https://direct.mit.edu/neco/article-pdf/8/1/182/813161/neco.1996.8.1.182.pdf
[tiled Cholesky]: http://eprints.ma.man.ac.uk/856/01/covered/MIMS_ep2007_122.pdf
[Adam Coates et al. 2010]: http://ai.stanford.edu/~acoates/papers/CoatesLeeNg_nips2010_dlwkshp_singlelayer.pdf
[bknyaz]: https://github.com/bknyaz/gabors

## Appendix: Python Setup, And All There Is to Machine Learning

I split the original Jupyter notebook file into two files: training (main.py) and prediction (save_predictions.py), and added the plotting file (plot_errors.py) which produces the figure above.

Workflow:

- Install [conda][anaconda]. It makes life so much easier, but also see this [emotional critique][anaconda-critique].

- My GPU card is GTX760, its computing capacity is 3.0 and that demands installing specific library versions provided in this [SO question][compute-capability-3.0]:
```console
conda create -n tf-gpu
conda activate tf-gpu
conda install tensorflow-gpu=1.12
conda install cudatoolkit=9.0
conda install -c anaconda scikit-learn
conda install -c conda-forge pudb
conda install -c conda-forge matplotlib
```

- To remove the environment some day:
```console
conda info --envs
conda deactivate
conda remove --name tf-gpu --all
```

- To test the code, run these commands:
```console
conda activate tf-gpu
python main.py
python save_predictions.py
python plot_errors.py
```

