# Semantic Segmentation with PyTorch
Experiments of different models given VOC2012 Datasets and SBD Datasets.
## Usage
```bash
git clone https://github.com/JamesHsu333/Thesis.git
cd Thesis
pip install -r requirements.txt
```
## Dataset
1. Download from
[VOC2012 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and
[SBD dataset](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
2. Configure your dataset path in ```dataloaders/dataloader.py```
### Data Preprocessing
The images of VOC2012 are 500x225 pixels. Use data augmentation such as Random Horizontal Flip, Random Scale Crop, Random Gaussian Blur, Normalize. Also, the images are resized to 513x513.
## Model Architecture
1. To check different models' architecture, open ```model_summary.py``` to choose different models. Then Run
```bash
python model_summary.py
```
-------
## Quickstart
1.  Created a ```params.json``` under the ```experiments``` directory. It sets the hyperparameters for the experiment which looks like
```Json
{
    "learning_rate": 0.007,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "batch_size": 8,
    "num_epochs": 25,
    "dropout_rate": 0.0,
    "num_channels": 32,
    "save_summary_steps": 100,
    "num_workers": 4,
    "base_size": 513,
    "crop_size": 513
}
```
2. Train your experiment. Run
```bash
python train.py
```
3. Display the results of the hyperparameters search in a nice format
```bash
python synthesize_results.py --parent_dir experiments
```
4. Evaluation on the test set Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set. Run
```bash
python evaluate.py --model_dir experiments/your_model_dirname
```
## Resources
* For more Project Structure details, please refer to [Deep Learning Project Structure](https://deeps.site/blog/2019/12/07/dl-project-structure/)
* Part of Code implementation refers from [jfzhang95/pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
* Part of Code implementation refers from [junfu1115/DANet](https://github.com/junfu1115/DANet)

## References
[[1]](https://arxiv.org/pdf/1703.02719) Chao Peng, Xiangyu Zhang, Gang Yu, Guiming Luo, Jian Sun. Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network, abs/1703.02719, 2017.

[[2]](https://arxiv.org/pdf/1802.02611.pdf) Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, abs/1802.02611, 2018.

[[3]](https://arxiv.org/pdf/1809.02983.pdf) Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang, Hanqing Lu. Dual Attention Network for Scene Segmentation, abs/1809.02983, 2019