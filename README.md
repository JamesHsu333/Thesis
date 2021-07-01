# DeepLabv3+ with PyTorch
An experiment of DeepLabv3+ given VOC2012 Datasets and SBD Datasets.
## Usage
```bash
git clone https://github.com/JamesHsu333/DeepLabv3plus.git
cd DeepLabv3plus
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
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 257, 257]           9,408
       BatchNorm2d-2         [-1, 64, 257, 257]             128
              ReLU-3         [-1, 64, 257, 257]               0
         MaxPool2d-4         [-1, 64, 129, 129]               0
            Conv2d-5         [-1, 64, 129, 129]           4,096
       BatchNorm2d-6         [-1, 64, 129, 129]             128
              ReLU-7         [-1, 64, 129, 129]               0
            Conv2d-8         [-1, 64, 129, 129]          36,864
       BatchNorm2d-9         [-1, 64, 129, 129]             128
             ReLU-10         [-1, 64, 129, 129]               0
           Conv2d-11        [-1, 256, 129, 129]          16,384
      BatchNorm2d-12        [-1, 256, 129, 129]             512
           Conv2d-13        [-1, 256, 129, 129]          16,384
      BatchNorm2d-14        [-1, 256, 129, 129]             512
             ReLU-15        [-1, 256, 129, 129]               0
       Bottleneck-16        [-1, 256, 129, 129]               0
           Conv2d-17         [-1, 64, 129, 129]          16,384
      BatchNorm2d-18         [-1, 64, 129, 129]             128
             ReLU-19         [-1, 64, 129, 129]               0
           Conv2d-20         [-1, 64, 129, 129]          36,864
      BatchNorm2d-21         [-1, 64, 129, 129]             128
             ReLU-22         [-1, 64, 129, 129]               0
           Conv2d-23        [-1, 256, 129, 129]          16,384
      BatchNorm2d-24        [-1, 256, 129, 129]             512
             ReLU-25        [-1, 256, 129, 129]               0
       Bottleneck-26        [-1, 256, 129, 129]               0
           Conv2d-27         [-1, 64, 129, 129]          16,384
      BatchNorm2d-28         [-1, 64, 129, 129]             128
             ReLU-29         [-1, 64, 129, 129]               0
           Conv2d-30         [-1, 64, 129, 129]          36,864
      BatchNorm2d-31         [-1, 64, 129, 129]             128
             ReLU-32         [-1, 64, 129, 129]               0
           Conv2d-33        [-1, 256, 129, 129]          16,384
      BatchNorm2d-34        [-1, 256, 129, 129]             512
             ReLU-35        [-1, 256, 129, 129]               0
       Bottleneck-36        [-1, 256, 129, 129]               0
           Conv2d-37        [-1, 128, 129, 129]          32,768
      BatchNorm2d-38        [-1, 128, 129, 129]             256
             ReLU-39        [-1, 128, 129, 129]               0
           Conv2d-40          [-1, 128, 65, 65]         147,456
      BatchNorm2d-41          [-1, 128, 65, 65]             256
             ReLU-42          [-1, 128, 65, 65]               0
           Conv2d-43          [-1, 512, 65, 65]          65,536
      BatchNorm2d-44          [-1, 512, 65, 65]           1,024
           Conv2d-45          [-1, 512, 65, 65]         131,072
      BatchNorm2d-46          [-1, 512, 65, 65]           1,024
             ReLU-47          [-1, 512, 65, 65]               0
       Bottleneck-48          [-1, 512, 65, 65]               0
           Conv2d-49          [-1, 128, 65, 65]          65,536
      BatchNorm2d-50          [-1, 128, 65, 65]             256
             ReLU-51          [-1, 128, 65, 65]               0
           Conv2d-52          [-1, 128, 65, 65]         147,456
      BatchNorm2d-53          [-1, 128, 65, 65]             256
             ReLU-54          [-1, 128, 65, 65]               0
           Conv2d-55          [-1, 512, 65, 65]          65,536
      BatchNorm2d-56          [-1, 512, 65, 65]           1,024
             ReLU-57          [-1, 512, 65, 65]               0
       Bottleneck-58          [-1, 512, 65, 65]               0
           Conv2d-59          [-1, 128, 65, 65]          65,536
      BatchNorm2d-60          [-1, 128, 65, 65]             256
             ReLU-61          [-1, 128, 65, 65]               0
           Conv2d-62          [-1, 128, 65, 65]         147,456
      BatchNorm2d-63          [-1, 128, 65, 65]             256
             ReLU-64          [-1, 128, 65, 65]               0
           Conv2d-65          [-1, 512, 65, 65]          65,536
      BatchNorm2d-66          [-1, 512, 65, 65]           1,024
             ReLU-67          [-1, 512, 65, 65]               0
       Bottleneck-68          [-1, 512, 65, 65]               0
           Conv2d-69          [-1, 128, 65, 65]          65,536
      BatchNorm2d-70          [-1, 128, 65, 65]             256
             ReLU-71          [-1, 128, 65, 65]               0
           Conv2d-72          [-1, 128, 65, 65]         147,456
      BatchNorm2d-73          [-1, 128, 65, 65]             256
             ReLU-74          [-1, 128, 65, 65]               0
           Conv2d-75          [-1, 512, 65, 65]          65,536
      BatchNorm2d-76          [-1, 512, 65, 65]           1,024
             ReLU-77          [-1, 512, 65, 65]               0
       Bottleneck-78          [-1, 512, 65, 65]               0
           Conv2d-79          [-1, 256, 65, 65]         131,072
      BatchNorm2d-80          [-1, 256, 65, 65]             512
             ReLU-81          [-1, 256, 65, 65]               0
           Conv2d-82          [-1, 256, 33, 33]         589,824
      BatchNorm2d-83          [-1, 256, 33, 33]             512
             ReLU-84          [-1, 256, 33, 33]               0
           Conv2d-85         [-1, 1024, 33, 33]         262,144
      BatchNorm2d-86         [-1, 1024, 33, 33]           2,048
           Conv2d-87         [-1, 1024, 33, 33]         524,288
      BatchNorm2d-88         [-1, 1024, 33, 33]           2,048
             ReLU-89         [-1, 1024, 33, 33]               0
       Bottleneck-90         [-1, 1024, 33, 33]               0
           Conv2d-91          [-1, 256, 33, 33]         262,144
      BatchNorm2d-92          [-1, 256, 33, 33]             512
             ReLU-93          [-1, 256, 33, 33]               0
           Conv2d-94          [-1, 256, 33, 33]         589,824
      BatchNorm2d-95          [-1, 256, 33, 33]             512
             ReLU-96          [-1, 256, 33, 33]               0
           Conv2d-97         [-1, 1024, 33, 33]         262,144
      BatchNorm2d-98         [-1, 1024, 33, 33]           2,048
             ReLU-99         [-1, 1024, 33, 33]               0
      Bottleneck-100         [-1, 1024, 33, 33]               0
          Conv2d-101          [-1, 256, 33, 33]         262,144
     BatchNorm2d-102          [-1, 256, 33, 33]             512
            ReLU-103          [-1, 256, 33, 33]               0
          Conv2d-104          [-1, 256, 33, 33]         589,824
     BatchNorm2d-105          [-1, 256, 33, 33]             512
            ReLU-106          [-1, 256, 33, 33]               0
          Conv2d-107         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-108         [-1, 1024, 33, 33]           2,048
            ReLU-109         [-1, 1024, 33, 33]               0
      Bottleneck-110         [-1, 1024, 33, 33]               0
          Conv2d-111          [-1, 256, 33, 33]         262,144
     BatchNorm2d-112          [-1, 256, 33, 33]             512
            ReLU-113          [-1, 256, 33, 33]               0
          Conv2d-114          [-1, 256, 33, 33]         589,824
     BatchNorm2d-115          [-1, 256, 33, 33]             512
            ReLU-116          [-1, 256, 33, 33]               0
          Conv2d-117         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-118         [-1, 1024, 33, 33]           2,048
            ReLU-119         [-1, 1024, 33, 33]               0
      Bottleneck-120         [-1, 1024, 33, 33]               0
          Conv2d-121          [-1, 256, 33, 33]         262,144
     BatchNorm2d-122          [-1, 256, 33, 33]             512
            ReLU-123          [-1, 256, 33, 33]               0
          Conv2d-124          [-1, 256, 33, 33]         589,824
     BatchNorm2d-125          [-1, 256, 33, 33]             512
            ReLU-126          [-1, 256, 33, 33]               0
          Conv2d-127         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-128         [-1, 1024, 33, 33]           2,048
            ReLU-129         [-1, 1024, 33, 33]               0
      Bottleneck-130         [-1, 1024, 33, 33]               0
          Conv2d-131          [-1, 256, 33, 33]         262,144
     BatchNorm2d-132          [-1, 256, 33, 33]             512
            ReLU-133          [-1, 256, 33, 33]               0
          Conv2d-134          [-1, 256, 33, 33]         589,824
     BatchNorm2d-135          [-1, 256, 33, 33]             512
            ReLU-136          [-1, 256, 33, 33]               0
          Conv2d-137         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-138         [-1, 1024, 33, 33]           2,048
            ReLU-139         [-1, 1024, 33, 33]               0
      Bottleneck-140         [-1, 1024, 33, 33]               0
          Conv2d-141          [-1, 256, 33, 33]         262,144
     BatchNorm2d-142          [-1, 256, 33, 33]             512
            ReLU-143          [-1, 256, 33, 33]               0
          Conv2d-144          [-1, 256, 33, 33]         589,824
     BatchNorm2d-145          [-1, 256, 33, 33]             512
            ReLU-146          [-1, 256, 33, 33]               0
          Conv2d-147         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-148         [-1, 1024, 33, 33]           2,048
            ReLU-149         [-1, 1024, 33, 33]               0
      Bottleneck-150         [-1, 1024, 33, 33]               0
          Conv2d-151          [-1, 256, 33, 33]         262,144
     BatchNorm2d-152          [-1, 256, 33, 33]             512
            ReLU-153          [-1, 256, 33, 33]               0
          Conv2d-154          [-1, 256, 33, 33]         589,824
     BatchNorm2d-155          [-1, 256, 33, 33]             512
            ReLU-156          [-1, 256, 33, 33]               0
          Conv2d-157         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-158         [-1, 1024, 33, 33]           2,048
            ReLU-159         [-1, 1024, 33, 33]               0
      Bottleneck-160         [-1, 1024, 33, 33]               0
          Conv2d-161          [-1, 256, 33, 33]         262,144
     BatchNorm2d-162          [-1, 256, 33, 33]             512
            ReLU-163          [-1, 256, 33, 33]               0
          Conv2d-164          [-1, 256, 33, 33]         589,824
     BatchNorm2d-165          [-1, 256, 33, 33]             512
            ReLU-166          [-1, 256, 33, 33]               0
          Conv2d-167         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-168         [-1, 1024, 33, 33]           2,048
            ReLU-169         [-1, 1024, 33, 33]               0
      Bottleneck-170         [-1, 1024, 33, 33]               0
          Conv2d-171          [-1, 256, 33, 33]         262,144
     BatchNorm2d-172          [-1, 256, 33, 33]             512
            ReLU-173          [-1, 256, 33, 33]               0
          Conv2d-174          [-1, 256, 33, 33]         589,824
     BatchNorm2d-175          [-1, 256, 33, 33]             512
            ReLU-176          [-1, 256, 33, 33]               0
          Conv2d-177         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-178         [-1, 1024, 33, 33]           2,048
            ReLU-179         [-1, 1024, 33, 33]               0
      Bottleneck-180         [-1, 1024, 33, 33]               0
          Conv2d-181          [-1, 256, 33, 33]         262,144
     BatchNorm2d-182          [-1, 256, 33, 33]             512
            ReLU-183          [-1, 256, 33, 33]               0
          Conv2d-184          [-1, 256, 33, 33]         589,824
     BatchNorm2d-185          [-1, 256, 33, 33]             512
            ReLU-186          [-1, 256, 33, 33]               0
          Conv2d-187         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-188         [-1, 1024, 33, 33]           2,048
            ReLU-189         [-1, 1024, 33, 33]               0
      Bottleneck-190         [-1, 1024, 33, 33]               0
          Conv2d-191          [-1, 256, 33, 33]         262,144
     BatchNorm2d-192          [-1, 256, 33, 33]             512
            ReLU-193          [-1, 256, 33, 33]               0
          Conv2d-194          [-1, 256, 33, 33]         589,824
     BatchNorm2d-195          [-1, 256, 33, 33]             512
            ReLU-196          [-1, 256, 33, 33]               0
          Conv2d-197         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-198         [-1, 1024, 33, 33]           2,048
            ReLU-199         [-1, 1024, 33, 33]               0
      Bottleneck-200         [-1, 1024, 33, 33]               0
          Conv2d-201          [-1, 256, 33, 33]         262,144
     BatchNorm2d-202          [-1, 256, 33, 33]             512
            ReLU-203          [-1, 256, 33, 33]               0
          Conv2d-204          [-1, 256, 33, 33]         589,824
     BatchNorm2d-205          [-1, 256, 33, 33]             512
            ReLU-206          [-1, 256, 33, 33]               0
          Conv2d-207         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-208         [-1, 1024, 33, 33]           2,048
            ReLU-209         [-1, 1024, 33, 33]               0
      Bottleneck-210         [-1, 1024, 33, 33]               0
          Conv2d-211          [-1, 256, 33, 33]         262,144
     BatchNorm2d-212          [-1, 256, 33, 33]             512
            ReLU-213          [-1, 256, 33, 33]               0
          Conv2d-214          [-1, 256, 33, 33]         589,824
     BatchNorm2d-215          [-1, 256, 33, 33]             512
            ReLU-216          [-1, 256, 33, 33]               0
          Conv2d-217         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-218         [-1, 1024, 33, 33]           2,048
            ReLU-219         [-1, 1024, 33, 33]               0
      Bottleneck-220         [-1, 1024, 33, 33]               0
          Conv2d-221          [-1, 256, 33, 33]         262,144
     BatchNorm2d-222          [-1, 256, 33, 33]             512
            ReLU-223          [-1, 256, 33, 33]               0
          Conv2d-224          [-1, 256, 33, 33]         589,824
     BatchNorm2d-225          [-1, 256, 33, 33]             512
            ReLU-226          [-1, 256, 33, 33]               0
          Conv2d-227         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-228         [-1, 1024, 33, 33]           2,048
            ReLU-229         [-1, 1024, 33, 33]               0
      Bottleneck-230         [-1, 1024, 33, 33]               0
          Conv2d-231          [-1, 256, 33, 33]         262,144
     BatchNorm2d-232          [-1, 256, 33, 33]             512
            ReLU-233          [-1, 256, 33, 33]               0
          Conv2d-234          [-1, 256, 33, 33]         589,824
     BatchNorm2d-235          [-1, 256, 33, 33]             512
            ReLU-236          [-1, 256, 33, 33]               0
          Conv2d-237         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-238         [-1, 1024, 33, 33]           2,048
            ReLU-239         [-1, 1024, 33, 33]               0
      Bottleneck-240         [-1, 1024, 33, 33]               0
          Conv2d-241          [-1, 256, 33, 33]         262,144
     BatchNorm2d-242          [-1, 256, 33, 33]             512
            ReLU-243          [-1, 256, 33, 33]               0
          Conv2d-244          [-1, 256, 33, 33]         589,824
     BatchNorm2d-245          [-1, 256, 33, 33]             512
            ReLU-246          [-1, 256, 33, 33]               0
          Conv2d-247         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-248         [-1, 1024, 33, 33]           2,048
            ReLU-249         [-1, 1024, 33, 33]               0
      Bottleneck-250         [-1, 1024, 33, 33]               0
          Conv2d-251          [-1, 256, 33, 33]         262,144
     BatchNorm2d-252          [-1, 256, 33, 33]             512
            ReLU-253          [-1, 256, 33, 33]               0
          Conv2d-254          [-1, 256, 33, 33]         589,824
     BatchNorm2d-255          [-1, 256, 33, 33]             512
            ReLU-256          [-1, 256, 33, 33]               0
          Conv2d-257         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-258         [-1, 1024, 33, 33]           2,048
            ReLU-259         [-1, 1024, 33, 33]               0
      Bottleneck-260         [-1, 1024, 33, 33]               0
          Conv2d-261          [-1, 256, 33, 33]         262,144
     BatchNorm2d-262          [-1, 256, 33, 33]             512
            ReLU-263          [-1, 256, 33, 33]               0
          Conv2d-264          [-1, 256, 33, 33]         589,824
     BatchNorm2d-265          [-1, 256, 33, 33]             512
            ReLU-266          [-1, 256, 33, 33]               0
          Conv2d-267         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-268         [-1, 1024, 33, 33]           2,048
            ReLU-269         [-1, 1024, 33, 33]               0
      Bottleneck-270         [-1, 1024, 33, 33]               0
          Conv2d-271          [-1, 256, 33, 33]         262,144
     BatchNorm2d-272          [-1, 256, 33, 33]             512
            ReLU-273          [-1, 256, 33, 33]               0
          Conv2d-274          [-1, 256, 33, 33]         589,824
     BatchNorm2d-275          [-1, 256, 33, 33]             512
            ReLU-276          [-1, 256, 33, 33]               0
          Conv2d-277         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-278         [-1, 1024, 33, 33]           2,048
            ReLU-279         [-1, 1024, 33, 33]               0
      Bottleneck-280         [-1, 1024, 33, 33]               0
          Conv2d-281          [-1, 256, 33, 33]         262,144
     BatchNorm2d-282          [-1, 256, 33, 33]             512
            ReLU-283          [-1, 256, 33, 33]               0
          Conv2d-284          [-1, 256, 33, 33]         589,824
     BatchNorm2d-285          [-1, 256, 33, 33]             512
            ReLU-286          [-1, 256, 33, 33]               0
          Conv2d-287         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-288         [-1, 1024, 33, 33]           2,048
            ReLU-289         [-1, 1024, 33, 33]               0
      Bottleneck-290         [-1, 1024, 33, 33]               0
          Conv2d-291          [-1, 256, 33, 33]         262,144
     BatchNorm2d-292          [-1, 256, 33, 33]             512
            ReLU-293          [-1, 256, 33, 33]               0
          Conv2d-294          [-1, 256, 33, 33]         589,824
     BatchNorm2d-295          [-1, 256, 33, 33]             512
            ReLU-296          [-1, 256, 33, 33]               0
          Conv2d-297         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-298         [-1, 1024, 33, 33]           2,048
            ReLU-299         [-1, 1024, 33, 33]               0
      Bottleneck-300         [-1, 1024, 33, 33]               0
          Conv2d-301          [-1, 256, 33, 33]         262,144
     BatchNorm2d-302          [-1, 256, 33, 33]             512
            ReLU-303          [-1, 256, 33, 33]               0
          Conv2d-304          [-1, 256, 33, 33]         589,824
     BatchNorm2d-305          [-1, 256, 33, 33]             512
            ReLU-306          [-1, 256, 33, 33]               0
          Conv2d-307         [-1, 1024, 33, 33]         262,144
     BatchNorm2d-308         [-1, 1024, 33, 33]           2,048
            ReLU-309         [-1, 1024, 33, 33]               0
      Bottleneck-310         [-1, 1024, 33, 33]               0
          Conv2d-311          [-1, 512, 33, 33]         524,288
     BatchNorm2d-312          [-1, 512, 33, 33]           1,024
            ReLU-313          [-1, 512, 33, 33]               0
          Conv2d-314          [-1, 512, 33, 33]       2,359,296
     BatchNorm2d-315          [-1, 512, 33, 33]           1,024
            ReLU-316          [-1, 512, 33, 33]               0
          Conv2d-317         [-1, 2048, 33, 33]       1,048,576
     BatchNorm2d-318         [-1, 2048, 33, 33]           4,096
          Conv2d-319         [-1, 2048, 33, 33]       2,097,152
     BatchNorm2d-320         [-1, 2048, 33, 33]           4,096
            ReLU-321         [-1, 2048, 33, 33]               0
      Bottleneck-322         [-1, 2048, 33, 33]               0
          Conv2d-323          [-1, 512, 33, 33]       1,048,576
     BatchNorm2d-324          [-1, 512, 33, 33]           1,024
            ReLU-325          [-1, 512, 33, 33]               0
          Conv2d-326          [-1, 512, 33, 33]       2,359,296
     BatchNorm2d-327          [-1, 512, 33, 33]           1,024
            ReLU-328          [-1, 512, 33, 33]               0
          Conv2d-329         [-1, 2048, 33, 33]       1,048,576
     BatchNorm2d-330         [-1, 2048, 33, 33]           4,096
            ReLU-331         [-1, 2048, 33, 33]               0
      Bottleneck-332         [-1, 2048, 33, 33]               0
          Conv2d-333          [-1, 512, 33, 33]       1,048,576
     BatchNorm2d-334          [-1, 512, 33, 33]           1,024
            ReLU-335          [-1, 512, 33, 33]               0
          Conv2d-336          [-1, 512, 33, 33]       2,359,296
     BatchNorm2d-337          [-1, 512, 33, 33]           1,024
            ReLU-338          [-1, 512, 33, 33]               0
          Conv2d-339         [-1, 2048, 33, 33]       1,048,576
     BatchNorm2d-340         [-1, 2048, 33, 33]           4,096
            ReLU-341         [-1, 2048, 33, 33]               0
      Bottleneck-342         [-1, 2048, 33, 33]               0
          ResNet-343  [[-1, 2048, 33, 33], [-1, 256, 129, 129]]               0
          Conv2d-344          [-1, 256, 33, 33]         524,288
     BatchNorm2d-345          [-1, 256, 33, 33]             512
            ReLU-346          [-1, 256, 33, 33]               0
     _ASPPModule-347          [-1, 256, 33, 33]               0
          Conv2d-348          [-1, 256, 33, 33]       4,718,592
     BatchNorm2d-349          [-1, 256, 33, 33]             512
            ReLU-350          [-1, 256, 33, 33]               0
     _ASPPModule-351          [-1, 256, 33, 33]               0
          Conv2d-352          [-1, 256, 33, 33]       4,718,592
     BatchNorm2d-353          [-1, 256, 33, 33]             512
            ReLU-354          [-1, 256, 33, 33]               0
     _ASPPModule-355          [-1, 256, 33, 33]               0
          Conv2d-356          [-1, 256, 33, 33]       4,718,592
     BatchNorm2d-357          [-1, 256, 33, 33]             512
            ReLU-358          [-1, 256, 33, 33]               0
     _ASPPModule-359          [-1, 256, 33, 33]               0
AdaptiveAvgPool2d-360           [-1, 2048, 1, 1]               0
          Conv2d-361            [-1, 256, 1, 1]         524,288
     BatchNorm2d-362            [-1, 256, 1, 1]             512
            ReLU-363            [-1, 256, 1, 1]               0
          Conv2d-364          [-1, 256, 33, 33]         327,680
     BatchNorm2d-365          [-1, 256, 33, 33]             512
            ReLU-366          [-1, 256, 33, 33]               0
         Dropout-367          [-1, 256, 33, 33]               0
            ASPP-368          [-1, 256, 33, 33]               0
          Conv2d-369         [-1, 48, 129, 129]          12,288
     BatchNorm2d-370         [-1, 48, 129, 129]              96
            ReLU-371         [-1, 48, 129, 129]               0
          Conv2d-372        [-1, 256, 129, 129]         700,416
     BatchNorm2d-373        [-1, 256, 129, 129]             512
            ReLU-374        [-1, 256, 129, 129]               0
         Dropout-375        [-1, 256, 129, 129]               0
          Conv2d-376        [-1, 256, 129, 129]         589,824
     BatchNorm2d-377        [-1, 256, 129, 129]             512
            ReLU-378        [-1, 256, 129, 129]               0
         Dropout-379        [-1, 256, 129, 129]               0
          Conv2d-380         [-1, 21, 129, 129]           5,397
         Decoder-381         [-1, 21, 129, 129]               0
================================================================
Total params: 59,344,309
Trainable params: 59,344,309
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.01
Forward/backward pass size (MB): 2486.63
Params size (MB): 226.38
Estimated Total Size (MB): 2716.02
----------------------------------------------------------------
```
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

## References
[[1]](https://arxiv.org/pdf/1802.02611.pdf) Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, abs/1802.02611, 2018.