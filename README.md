# MaskReco
一个基于YoloV5的口罩识别项目+GUI


## 使用方式
1. 安装依赖
```
pip install -r requirements.txt 
```
2. 运行 ***PyqtGUI.py*** 文件
```
python PyqtGUI.py
```


best1、2、3.pt为模型文件，如有需要自行替换。


## 注释
对不起，由于我换了环境，在整理老项目代码并传到github上时，发现这个项目里缺少了 ***requirements.txt*** 文件，所以我用 ```pipreqs ./ --encoding=utf-8``` 重新生成的，但是大部分依赖，我的老环境中都卸掉了，所以版本并不完全对应。<br>

如果你不能运行请重新安装 ```pip install -r requirements2.txt ``` 的依赖，并手动补齐缺失的库。<br>
但这样做之前也请事先排查好torch版本的问题，
再次抱歉！！！！


<br>
<br>
<br>
<br> 
下面是曾经的拙劣文章(已过时)

---

## 目标检测:
目标检测是计算机视觉和数字图像处理的一个热门方向，广泛应用于无人驾驶、智能视频监控、工业检测、航空航天等诸多领域，通过计算机视觉减少对人力资本的消耗，具有重要的现实意义。 因此，目标检测也就成为了近年来理论和应用的研究热点，它是图像处理和计算机视觉学科的重要分支，也是智能监控系统的核心部分，同时目标检测也是泛身份识别领域的一个基础性的算法，对后续的人脸识别、步态识别、人群计数、实例分割等任务起着至关重要的作用。

------------


## YOLOv5简介:
YOLOV4出现之后不久，YOLOv5横空出世。YOLOv5在YOLOv4算法的基础上做了进一步的改进，检测性能得到进一步的提升。YOLOv5在COCO数据集上面的测试效果非常不错。工业界也往往更喜欢使用这些方法，而不是利用一个超级复杂的算法来获得较高的检测精度。
YOLOv5是一种单阶段目标检测算法，速度与精度都得到了极大的性能提升。主要的改进思路如下所示：
1. 输入端：在模型训练阶段，提出了一些改进思路，主要包括Mosaic数据增强、自适应锚框计算、自适应图片缩放。
2. 基准网络：融合其它检测算法中的一些新思路，主要包括：Focus结构与CSP结构。
3. Neck网络：目标检测网络在BackBone与最后的Head输出层之间往往会插入一些层，Yolov5中添加了FPN+PAN结构。
4. Head输出层：输出层的锚框机制与YOLOv4相同，主要改进的是训练时的损失函数GIOU_Loss，以及预测框筛选的DIOU_nms。


**YOLOv5S模型的网络架构：**

[![](https://s1.ax1x.com/2022/07/07/jdByjK.jpg)](https://s1.ax1x.com/2022/07/07/jdByjK.jpg)

Yolov5s网络是Yolov5系列中深度最小，特征图的宽度最小的网络。Yolov5m、Yolov5l、Yolov5x 都是在此基础上不断加深，不断加宽。

------------


## YOLOV5目录结构：
下载源码后解压可以看到如下目录：

![](https://s1.ax1x.com/2022/07/07/jdD85d.png)
其中，train.py这个文件也是我们接下来训练yolo模型需要用到的启动文件。
requirement.txt 中有我们所需要的的全部依赖,采用pip安装。
```python
pip install -r requirements.txt #安装完依赖后准备工作完成
```
**每个文件作用:**
```python
YOLOv5
|   detect.py  #检测脚本
|   hubconf.py  # PyTorch Hub相关代码
|   LICENSE    # 版权文件
|   README.md  #README  markdown 文件
|   requirements.txt   #项目所需的安装包列表
|   sotabench.py    #COCO 数据集测试脚本
|   test.py         #模型测试脚本
|   train.py         #模型训练脚本
|   tutorial.ipynb   #Jupyter Notebook演示代码
|-- data
|    |   coco.yaml           #COCO 数据集配置文件
|    |   coco128.yaml        #COCO128 数据集配置文件
|    |   hyp.finetune.yaml   #超参数微调配置文件
|    |   hyp.scratch.yaml    #超参数启始配置文件
|    |   voc.yaml            #VOC数据集配置文件
|    |---scripts
|             get_coco.sh   # 下载COCO数据集shell命令
|             get_voc.sh   # 下载VOC数据集shell命令
|-- inference
|    |   images         #示例图片文件夹
|              bus.jpg
|              zidane.jpg
|-- models
|    |   common.py         #模型组件定义代码
|    |   experimental.py   #实验性质的代码
|    |   export.py         #模型导出脚本
|    |   yolo.py           # Detect 及 Model构建代码
|    |   yolo5l.yaml       # yolov5l 网络模型配置文件
|    |   yolo5m.yaml       # yolov5m 网络模型配置文件
|    |   yolo5s.yaml       # yolov5s 网络模型配置文件
|    |   yolo5x.yaml       # yolov5x 网络模型配置文件
|    |   __init__.py       
|    |---hub
|            yolov3-spp.yaml
|            yolov3-fpn.yaml
|            yolov3-panet.yaml
|-- runs    #训练结果
|    |--exp0
|    |   |	events.out.tfevents.
|    |   |  hyp.yaml
|    |   |  labels.png
|    |   |  opt.yaml
|    |   |  precision-recall_curve.png
|    |   |  results.png
|    |   |  results.txt
|    |   |  test_batch0_gt.jpg
|    |   |  test_batch0_pred.jpg
|    |   |  test_batch0.jpg
|    |   |  test_batch1.jpg
|    |   |  test_batch2.jpg
|    |   |--weights
|    |             best.pt   #所有训练轮次中最好权重
|    |             last.pt   #最近一轮次训练权重
|-- utils
|    |   activations.py   #激活函数定义代码  
|    |   datasets.py      #Dataset 及Dataloader定义代码
|    |   evolve.py        #超参数进化命令 
|    |   general.py       #项目通用函数代码  
|    |   google_utils.py  # 谷歌云使用相关代码
|    |   torch_utils.py   # torch工具辅助类代码 
|    |   __init__.py   # torch工具辅助类代码 
|    |---google_app_engine
|               additional_requirements.txt
|				app.yaml
|				Dockerfile
|-- VOC   #数据集目录
|    |--images  #数据集图片目录
|    |    |--train  # 训练集图片文件夹
|    |    |         000005.jpg
|    |    |         000007.jpg
|    |    |         000009.jpg
|    |    |         0000012.jpg
|    |    |         0000016.jpg
|    |    |         ...
|    |    |--val    # 验证集图片文件夹 
|    |    |         000001.jpg
|    |    |         000002.jpg
|    |    |         000003.jpg
|    |    |         000004.jpg
|    |    |         000006.jpg
|    |    |         ...       
|    |--labels  #数据集标签目录
|    |    train.cache
|    |    val.cache
|    |    |--train  # 训练标签文件夹
|    |    |         000005.txt
|    |    |         000007.txt
|    |    |         ...
|    |    |--val    # 验证集图片文件夹 
|    |    |         000001.txt
|    |    |         000002.txt
|    |    |         ...    
|-- weights
    dwonload_weights.sh   #下载权重文件命令
    yolov5l.pt  #yolov5l 权重文件
    yolov5m.pt  #yolov5m 权重文件
    yolov5s.mlmodel  #yolov5s 权重文件(Core M格式)
    yolov5s.onnx  #yolov5s 权重文件(onnx格式)
    yolov5s.torchscript  #yolov5s 权重文件(torchscript格式)
    yolov5x.pt  #yolov5x 权重文件
```

------------


## 模型训练过程：

使用环境：Python3.8+torch1.8.1+cuda11.1+pycharm
（注：cuda的安装版本取决于显卡类型）


**1.数据集的标注：**

python打开labelimg这个软件进行标注。
```python
python labelimg.py
```
数据格式建议选择VOC,后期再转换成 yolo格式。
( VOC会生成 xml 文件,可以灵活转变为其他模型所需格式)

![](https://s1.ax1x.com/2022/07/07/jdDgx0.png)

本次训练标注两个标签，佩戴口罩为 mask，未佩戴口罩为 face。

在根目录下建立一个VOCData文件夹，再建立两个子文件，其中，jpg文件放置在VOCData/images下，xml放置在VOCData/Annotations中。（这一步根据个人随意，因为在训练时需要创建配置文件指定模型训练集的目录）



**2.数据集的训练：**

①、在项目根目录下文件夹下新建mask_data.yaml配置文件，添加如下内容：
（根据个人情况修改）
![](https://s1.ax1x.com/2022/07/07/jdrlQ0.png)
其中：
path：项目的根目录
train：训练集与path的相对路径
val：验证集与path的相对路径
nc：类别数量，2个
names：类别名字
(上一步中标注好的训练集，可以按照想要比例划分为训练和验证集，也可以不划分填同一路径。)

②、修改启动文件 train.py：

打开train.py,其相关参数如下：

![](https://s1.ax1x.com/2022/07/07/jdraWR.png)
其中：
weights：权重文件路径
cfg：存储模型结构的配置文件
data：存储训练、测试数据的文件（上一步中自己创建的那个.yaml）
epochs：训练过程中整个数据集的迭代次数
batch-size：训练后权重更新的图片数
img-size：输入图片宽高。
device：选择使用GPU还是CPU
workers：线程数，默认是8
```python
#输入命令开始训练：
python train.py --weights data/yolov5s.pt --cfg models/yolov5s.yaml --data data/mask_data.yaml --epoch 100 --batch-size 8 --device 0

```
③、等待慢慢跑完

![](https://s1.ax1x.com/2022/07/07/jdyAUg.png)

------------


## 模型结果数据呈现：
1.数据集的分布：
![](https://s1.ax1x.com/2022/07/07/jdy1VU.jpg)
mask的照片约有2000张，face的照片约有2500张。

2.损失函数和准确率：
![](https://s1.ax1x.com/2022/07/07/jdyYG9.png)
可以看到随着训练的进行，以不同方式呈现的损失函数呈明显下降趋势，准确率呈上升趋势。

3.置信度与准确率：
![](https://s1.ax1x.com/2022/07/07/jdyfqf.png)

置信度在0.6以上时，准确率接近80%。

------------

## GUI编程：

编写GUI界面，方便对权重文件进行一个替换，对图片和视频进行一个监测，以及调用摄像头进行实时监测。

呈现效果：
![](https://s1.ax1x.com/2022/07/07/jd6CW9.png)
![](https://s1.ax1x.com/2022/07/07/jd6PzR.png)
![](https://s1.ax1x.com/2022/07/07/jd6FQ1.png)

检测结果示意：
![](https://s1.ax1x.com/2022/07/07/jd6KWd.jpg)



------------


UI代码：
```python
class MainWindow(QTabWidget):

    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('Qust-口罩识别')
        self.resize(1200, 900)
        # 图片读取进程
        self.output_size = 500
        self.img2predict = ""
        self.device = '0'# todo 在这里设置使用 CPU还是GPU（0是GPU）
        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        #best123.pt 训练好的口罩模型
        self.model = self.model_load(weights="best2.pt",
                                     device=self.device)
        self.initUI()
        self.reset_vid()

    '''
    ***模型初始化***
    '''
    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成!")
        return model

    '''
    ***界面初始化***
    '''
    def initUI(self):
        # 图片检测子界面
        font_title = QFont('幼圆', 16)
        font_main = QFont('幼圆', 20)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片检测")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.png"))
        self.right_img.setPixmap(QPixmap("images/UI/right.png"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("选择图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(87,24,138);}"
                                    "QPushButton{background-color:rgb(46,169,223)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:20px 20px}"
                                    "QPushButton{margin:15px 150px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(87,24,138);}"
                                     "QPushButton{background-color:rgb(46,169,223)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:20px 20px}"
                                     "QPushButton{margin:15px 150px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # 视频识别子界面
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("视频检测")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/up.png"))
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("打开摄像头")
        self.mp4_detection_btn = QPushButton("打开视频文件")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(87,24,138);}"
                                                "QPushButton{background-color:rgb(46,169,223)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:15px 5px}"
                                                "QPushButton{margin:5px 150px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(87,24,138);}"
                                             "QPushButton{background-color:rgb(46,169,223)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:15px 5px}"
                                             "QPushButton{margin:5px 150px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(87,24,138);}"
                                        "QPushButton{background-color:rgb(46,169,223)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:15px 5px}"
                                        "QPushButton{margin:5px 150px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        # 选择权重文件子界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        self.about_title2 = QLabel('当前在使用GPU计算')
        self.about_title = QLabel('当前的权重文件是：')
        self.about_title.setFont(QFont('幼圆', 18))
        self.about_title2.setFont(QFont('幼圆', 18))
        path="best2.pt"
        self.label_patch = QLabel()
        self.label_patch.setText(path)
        self.label_patch.setFont(QFont('黑体', 16))
        self.label_patch.setWordWrap(1)
        self.about_title.setAlignment(Qt.AlignCenter)
        self.about_title2.setAlignment(Qt.AlignCenter)
        self.label_patch.setAlignment(Qt.AlignCenter)
        self.choose_button = QPushButton("选择权重文件")
        self.choose_button.setFont(font_main)
        self.choose_button.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(87,24,138);}"
                                        "QPushButton{background-color:rgb(46,169,223)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:15px 5px}"
                                        "QPushButton{margin:150px}"
                                        "QPushButton{margin-top:25px}")
        self.choose_button2 = QPushButton("点击切换CPU/GPU计算")
        self.choose_button2.setFont(font_main)
        self.choose_button2.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(87,24,138);}"
                                        "QPushButton{background-color:rgb(46,169,223)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:15px 5px}"
                                        "QPushButton{margin:150px}"
                                        "QPushButton{margin-top:5px}")
        about_layout.addWidget(self.about_title)
        about_layout.addWidget(self.label_patch)
        about_layout.addWidget(self.choose_button)
        mid_img_layout.addStretch(0)
        about_layout.addWidget(self.about_title2)
        about_layout.addWidget(self.choose_button2)
        about_widget.setLayout(about_layout)
        self.choose_button.clicked.connect(self.chpath)
        self.choose_button2.clicked.connect(self.chpath2)
        #设置tab栏
        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片识别')
        self.addTab(vid_detection_widget, '视频识别')
        self.addTab(about_widget, '选择权重文件')


```




