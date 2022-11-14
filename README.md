# CNCeleb

## Prepare

### Install CUDA-8.0

make sure cuda-8.0 is installed

### Install gcc-6.1.0

make sure gcc-6.1.0 is installed

### Install portaudio

install portaudio V19

### Establish Environment

#### install mxnet manually

```bash
# create virtual environment
conda create -n cnceleb python=2.7.12
source activate cnceleb
# install torch==0.4.0
# notice the difference between 'cp27mu' & 'cp27m'
wget https://download.pytorch.org/whl/cu80/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl
pip install torch-0.4.0-cp27-cp27mu-linux_x86_64.whl
pip install numpy==1.14.3
pip install scipy==1.0.1
pip install opencv_contrib_python
pip install python_speech_features==0.6
pip install llvmlite==0.31.0
pip install protobuf==3.8.0
pip install cython
pip install keras==2.2.5
pip install tensorflow-gpu==1.14.0
pip install pyaudio
pip install MxNet
pip install future
pip install sklearn
pip install scikit-image
pip install easydict
pip install pandas
pip install xlrd==1.2.0
pip install imageio==2.6.1
pip install librosa==0.7.2
# make RetinaFace
cd RetinaFace
cd RetinaFace/insightface/RetinaFace_linux/
make
```
## Environments
### python2
python version 2.7.12

提供了pip freeze导出的requirements.txt

### python3
python version 3.8.13

requirements:
1)	requests
2)	tqdm
3)	you-get

## Path to modify

需要修改以下位置的路径以成功运行：

| filename                      | line       |
| ----------------------------- | ---------- |
| run.sh                        | line 6~17  |
| getpoi/getpoi.py              | line 25    |
| speaker-Diarization/common.py | line 4~6   |
| videoprocess/common.py        | line 11~19 |
| tools/download_video.py        | line 65 |

## Model to download

下载 [20180402-114759/](https://cloud.tsinghua.edu.cn/d/8c454b96e9ea48698845) 到 `getpoi/facenet_code/20180402-114759/`

## Total Pipeline

1. 手动下载视频，生成对应的json文件记录视频原始的url和要保存的文件名信息
2. 使用`run.sh`处理视频
3. 人工审核处理后的视频
4. 发布数据

## 多人协作

收集数据是一项费时费力的工作，我们通常由多名同学一起进行收集，因此一个规范的多人协作流程是非常有必要的。我们推荐以下处理流程：

1. 开会决定需要收集的数量，并分配到每个人
2. 决定要收集的明星名单，确保和以往的名单没有交集，每个人认领自己任务数量的POI
3. 每人手动收集POI对应的视频，要求：
   1. 场景种类尽可能多
   2. 每个场景下的视频数量尽可能多
   3. 单个POI的不同场景下的视频相互无交集
   4. 单个POI视频总时长4h左右
4. 分批次调用run.sh处理收集到的视频，需要注意：
   1. 本repo并没有很有效的并行处理方法，一般采取复制整个repo进行多个job的并行处理
   2. 通常一个job占用的显存不超过3G
5. 汇总已处理的视频，每个视频分配两名审核人员依次进行审核
6. 汇总已审核的视频，进行抽查和发布

根据经验，预估每个人每天可以较高质量地收集7~10个POI的视频，视频处理所需的时长约为视频时长的一倍。

## Demo

本目录下的`demo/`文件夹给出了对两个POI进行全流程操作的一个实例，你可以跟着本文给出的代码在`demo/part0/`中一步步操作，并对得到的结果与`demo/part0_example/`中的示例进行对比以确保自己处于正确的道路上。

由于文件太大，请从链接中下载[part0_example](https://cloud.tsinghua.edu.cn/f/5cc4c5e695064b22915c/)并解压为`demo/part0_example/`

特别关注：我们只需要运行【在本demo中，你只需要运行：】后面给出的代码即可运行这个demo

### step -1: 创建正确的文件夹结构

`demo/part0/`中展示了正确的结构应该是什么样子的。确切的说，在你真正运行的过程中，只有POI的名字和genre可能会与`demo/part0/`有所不同，而其他的部分皆应保持一致。

### step 0 : 手动下载视频

在所有处理之前，我们需要手动下载视频，并构造一个包含视频信息的json文件，其内容必须包括`url`和`savefile`两项，一个示例如下：
```
[
    {
        "url": "http://www.bilibili.com/video/av331485600",
        "savefile": "BV1UA411u7qa"
    }
]
```

这可能听起来很麻烦，幸运的是，使用`tools/download_video.py`中提供的爬虫可以快速地爬取b站搜索得到的视频并生成对应的json，也可以根据json直接下载视频！

但需要注意的是，此脚本应当运行在**python3**环境下（与step 1需要的环境不同），同时必须遵循以下步骤：

- 必须在改写代码后手动调用脚本进行爬取
- 在爬取后得到json文件，必须人工审核并删除其中不符合需求的视频
- 在人工审核完json文件后，才能改写代码并调用脚本下载对应的视频

方便起见，本repo已经提供了人工审核后的json文件，并且处于数据量的考虑已经删除了大部分的视频，以使得demo的运行更加快速。

在本demo中，你只需要运行：

```shell
cd tools/
python download_video.py
```

### step 1 : 调用run.sh

注意，该步骤运行在**python2**环境下
`run.sh` 是对视频进行各种处理的最关键脚本，它串联起了整个处理流程中的绝大部分，如果感兴趣的话，本文最后给出了它对视频进行处理的pipeline（强烈推荐去看一下哦）。

如果顺利的话，在这个步骤中我们需要做的事情并不多。但如果运行中出了问题，那么就需要你自己来找找问题所在了~

在本demo中，你只需要运行：
```shell
sh run.sh
```

### step 2 : 审核切片

在`run.sh`运行完毕后，使用`tools/generate_splited_json.py`根据`run.sh`运行得到的最终结果生成所有视频切片的json文件。文件名为final_data.json，以在名称上对应`data/final_data/`。但需要注意的是，此脚本应当运行在**python3**环境下**（与step 1需要的环境不同）**。

然后，我们需要人工地审核json文件中给出的所有视频切片，并对其内容进行修改。通过修改json文件，我们就可以决定哪些视频需要删除，这省去了直接操作视频的麻烦。

审核操作通常需要两轮，即A先对`final_data.json`进行一遍审核得到`checked_A.json`，再由B对`checked_A.json`进行一遍审核得到`checked.json`。

通常我们使用`tools/check_json_utils/`中给出的本地网页工具来进行审核。以A的操作为例，具体方法是：

1. 将`final_data.json`和`cncheck.html`和`jsSaver.js`放在所有`.mp4`文件的同目录下
2. 通过浏览器打开cncheck.html ，点击【选择文件】并选择对应的`final_data.json`
3. 点击【init】按钮
4. 根据视频内容是否符合要求，对此视频分别使用【keep and next】【delet and next】进行保留或者删除；
   1. 如果保留，可以选择是否修改场景类型
   2. 如果删除，不必需要删除理由
5. 在审核完json文件中所有视频之后，点击【save current】保存审核后的结果。
6. 手动将保存得到的json文件修改为合适的文件名并保存在合适位置。

在本demo中，你只需要运行：
```
python generate_splited_json.py \
    --metadata-path /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/data/metadata/ \
    --split-path /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/logs/sysp_result/ \
    --data-path /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/data/final_data/

cp /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/data/metadata/超级小桀/interview/final_data.json /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/data/metadata/超级小桀/interview/checked.json
cp /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/data/metadata/超级小桀/livebroadcast/final_data.json /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/data/metadata/超级小桀/livebroadcast/checked.json

cp /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/data/metadata/寅子/interview/final_data.json /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/data/metadata/寅子/interview/checked.json
cp /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/data/metadata/寅子/livebroadcast/final_data.json /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/data/metadata/寅子/livebroadcast/checked.json
```

### step 3 : 发布数据

在审核完毕所有json后，我们就可以根据最终保留的视频文件信息来生成可以直接发布的数据，这一步骤由`tools/release_data.py`完全包含，我们只需要传入合适的参数，请阅读上述python文件后进行调用。但需要注意的是，此脚本应当运行在**python3环境**下**（与step 1需要的环境不同）**。

在本demo中，你只需要运行：
```
python release_data.py \
    --work-dir /work9/cchen/project/CNCeleb/cnceleb_data_collector/demo/part0/ \
    --video-folder data/final_data/ \
    --json-folder data/metadata/ \
    --startid 20000 \
    --min-duration 1.5 \
    --max-duration 90
```

## Pipeline of run.sh

### Stage 0: getpoi

下载并筛选POI的图片

input: 只有一列人名的excel表 name_list.xls

output: 不超过10张jpg图片，保存于${img_path}/name/name-seq.jpg（序号不一定连续）

pipeline:
- 通过爬虫从百度图片下载120张以POI为关键词搜索出来的图片
- 使用facenet筛选出10张图片
    - 读取并使用mtcnn检测人脸，删去有多个人脸或者没有人脸的图片
    - 截取检测出的人脸部分并resize为固定大小的图片
    - 使用facenet计算图片之间的距离
    - 选择离其他图片最近的10张，删除其他的
    - 保留的是原始图片

### Stage 1: transcoding

转码后确保mp4的帧率为25

input: 手动下载的已转为mp4格式的video

output: fps==25的mp4格式的video，命名为${output_video_path}/name/genre/genre-seq.mp4（序号连续）

pipeline:
- 如果fps=25，直接复制文件到指定文件夹，否则使用moviepy读取并写出为fps=25的mp4文件

### Stage 2: syncnent

计算音频和口唇的一致性

input: stage0得到的image和stage1得到的video

output: ${syncnet_output_dir}/name/genre/genre-seq.txt，内容为：
- 第一行：置信度阈值
- 之后行：帧号 置信度

pipeline:
- 使用insightface注册POI人脸图片（原始图片经facenet处理后到标准尺寸）
- 使用ffmpeg提取音轨为单独的wav文件
- 使用opencv读取视频内容
- 使用RetainFace检测人脸
- 对检测到的人脸用insightface进行验证，通过验证后使用opencv的Tracker追踪
- 当Tracker追踪结束后取得这一部分视频，通过syncnet与wav文件对应部分进行一致性检测，得到置信度
- 将置信度写入输出的txt文件

### Stage 3: speakerDiarization

计算音频中每一帧和目标人声的相似度

input: stage1输出的video

output: ${sysp_output_dir}/name/genre/genre-seq.txt，内容为：
- 第一行：长度（帧） overlap
- 其他行：余弦距离

pipeline:
- ffmpeg提取wav文件并计算spectogram，取幅值
- 对上一步的wav文件，用sox取stage2输出标注的部分，计算POI音频特征
- 对每一帧的音频计算特征，并与POI音频特征计算距离
- 写入输出文件

### Stage 4: speakerDiarization predict

merge stage2 & stage3 一种分数融合方法

input: stage2 和 stage3 的输出 txt

output: 目标POI可用数据在转码后视频中的时间段，覆盖了stage3的输出 ${sysp_output_dir}/name/genre/genre-seq.txt，内容为：
- 起始时间 结束时间（时:分:秒:帧）

### Stage 5：split

input: stage1输出的视频，stage4输出的txt

output: 按照stage4输出的txt切分转码后的视频
