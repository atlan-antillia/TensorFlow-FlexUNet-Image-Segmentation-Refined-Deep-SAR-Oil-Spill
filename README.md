<h2>TensorFlow-FlexUNet-Image-Segmentation-Refined-Deep-SAR-Oil-Spill  (2025/12/30)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>SAR Oil Spill (SOS)</b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and 
<a href="https://zenodo.org/records/15298010">Refined Deep-SAR Oil Spill (SOS) dataset </a> on zenodo.org.
<br><br>
<hr>
<b>Actual Image Segmentation for Refined Deep-SAR Oil Spill (SOS) Images of 256x256 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks, but they lack precision in certain areas.
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/images/palsar_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/masks/palsar_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test_output/palsar_2.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/images/palsar_56.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/masks/palsar_56.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test_output/palsar_56.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/images/palsar_108.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/masks/palsar_108.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test_output/palsar_108.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was obtained from <br><br>
<a href="https://zenodo.org/records/15298010">Refined Deep-SAR Oil Spill (SOS) dataset </a> on zenodo.org.

<br><br>
This is the improved version of the Deep-SAR Oil Spill (SOS) datase [1]. 
To enhance training data quality, segmentation masks with significant annotation errors were manually corrected.<br> 
Approximately 38% of the training masks and 50% of the validation masks were revised based on visual inspection and annotation consistency. <br>
These refinements aimed to reduce label noise and improve model performance during training and evaluation.
<br>
<br>
[1] Q. Zhu, Y. Zhang, Z. Li, X. Yan, Q. Guan, Y. Zhong,  L. Zhang, and D. Li, <br>
“Oil Spill Contextual and Boundary-Supervised Detection Network Based on Marine SAR Images,<br>
” IEEE Transactions on Geoscience and Remote Sensing, 2021.DOI: 10.1109/TGRS.2021.31<br>

<br>
<b>Citation</b><br>
Citation
Authors. (2025). Refined Deep-SAR Oil Spill (SOS) dataset [Data set]. Zenodo.<br>
<a href="https://doi.org/10.5281/zenodo.15298010"> https://doi.org/10.5281/zenodo.15298010</a>
<br>
<br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/legalcode">Creative Commons Attribution 4.0 International</a>
<br>
<br>
<h3>
2 SOS ImageMask Dataset
</h3>
 If you would like to train this SOS Segmentation model by yourself,
please down load the following two zip files in the web site <a href="https://zenodo.org/records/15298010">Refined Deep-SAR Oil Spill (SOS) dataset</a><br> 

<a  href="https://zenodo.org/records/15298010/files/images.zip?download=1">images.zip</a><br>
<a  href="https://zenodo.org/records/15298010/files/masks.zip?download=1">masks.zip</a><br>
, and expand the downloaded in a working directory.<br>
The folder structure of the <b>SOS</b> dataset is the following.<br> 
<pre>
./SOS
├─images
│   ├─train
│         ├─palsar_0.png
          ...
│         └─sentinel_3353.png
│   └─val
│         ├─palsar_0.png
           ...
│         └─sentinel_838.png
└─masks
    ├─train
    │     ├─palsar_0.png
           ...
    │     └─sentinel_3353.png
    └─val
          ├─palsar_0.png
          ...
          └─sentinel_838.png
</pre>
We used a Python script <a href="./generator/split_master.py">split_master.py</a> to split the master train  into
<b>train</b> and <b>test</b> subsets, and rearranged the original images and masks  folders to take the following structure.<br>
<pre>
./dataset
└─SOS
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>SOS Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/SOS/SOS_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/SOS/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/SOS/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained SOS TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/SOS/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/SOS and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 256
image_height   = 256
image_channels = 3
input_normalize = True
normalization  = False

num_classes    = 2

base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8

dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for SOS 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;SOS 1+1
rgb_map = {(0,0,0):0,  (255, 255, 255):1, }       
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/SOS/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (12,13,14)</b><br>
<img src="./projects/TensorFlowFlexUNet/SOS/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (26,27,28)</b><br>
<img src="./projects/TensorFlowFlexUNet/SOS/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 28 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/SOS/asset/train_console_output_at_epoch28.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/SOS/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/SOS/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/SOS/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/SOS/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/SOS</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for SOS.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/SOS/asset/evaluate_console_output_at_epoch28.png" width="880" height="auto">
<br><br>Image-Segmentation-SOS

<a href="./projects/TensorFlowFlexUNet/SOS/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this SOS/test was not low, and dice_coef_multiclass  not high as shown below.
<br>
<pre>
categorical_crossentropy,0.1787
dice_coef_multiclass,0.9047
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/SOS</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for SOS.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/SOS/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/SOS/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/SOS/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Refined Deep-SAR Oil Spill (SOS) Images of 256x256 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks, but they lack precision in certain areas.
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/images/palsar_7.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/masks/palsar_7.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test_output/palsar_7.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/images/palsar_56.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/masks/palsar_56.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test_output/palsar_56.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/images/palsar_108.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/masks/palsar_108.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test_output/palsar_108.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/images/palsar_309.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/masks/palsar_309.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test_output/palsar_309.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/images/palsar_172.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/masks/palsar_172.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test_output/palsar_172.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/images/palsar_333.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test/masks/palsar_333.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SOS/mini_test_output/palsar_333.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Oil Spill Detection using Convolutional Neural Networks and Sentinel-1 SAR Imagery</b><br>
Eleftheria Kalogirou, Konstantinos Christofi, Despoina Makri, Muhammad Amjad Iqbal, Valeria La Pegna,<br>
Marios Tzouvaras, Christodoulos Mettas, Diofantos Hadjimitsis<br>
<a href="https://isprs-archives.copernicus.org/articles/XLVIII-G-2025/757/2025/isprs-archives-XLVIII-G-2025-757-2025.pdf">
https://isprs-archives.copernicus.org/articles/XLVIII-G-2025/757/2025/isprs-archives-XLVIII-G-2025-757-2025.pdf</a>
<br>
<br>
<b>2. Oil spill detection and classification through deep learning and tailored data augmentation</b><br>
Ngoc An Bui,  Youngon Oh,  Impyeong Lee<br>
<a href="https://www.sciencedirect.com/science/article/pii/S1569843224001997">https://www.sciencedirect.com/science/article/pii/S1569843224001997</a>
<br>
<br>
<b>3. Oil spill detection by imaging radars: Challenges and pitfalls</b><br>
Werner Alpers , Benjamin Holt, Kan Zeng <br>
<a href="https://www.sciencedirect.com/science/article/pii/S0034425717304145">https://www.sciencedirect.com/science/article/pii/S0034425717304145</a>
<br>
<br>
<b>4.  Detection of Oil Spill in SAR Image Using an Improved DeepLabV3+</b><br>
Jiahao Zhang, Pengju Yang, and Xincheng Ren <br>
<a href="https://www.mdpi.com/1424-8220/24/17/5460">https://www.mdpi.com/1424-8220/24/17/5460</a>
<br>
<br>
<b>5.  Oil-Spill-Detection</b><br>
Harsha0112<br>
<a href="https://github.com/Harsha0112/Oil-Spill-Detection">https://github.com/Harsha0112/Oil-Spill-Detection</a>
<br>
<br>
<b>6. TensorFlow-FlexUNet-Image-Segmentation-Oil-Spill</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Oil-Spill">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Oil-Spill
</a>
<br>
<br>
