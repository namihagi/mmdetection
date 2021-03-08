## direstory structure

~~~
/mmdetection
  |-- data
  |  |-- coco
  |  |  |-- train2017
  |  |  |-- val2017
  |  |  |-- test2017
  |  |  |-- annotations
  |  |  
  |  |-- VOCdevkit
  |     |-- VOC2007
  |     |-- VOC2012
  |
  |-- docker
  |-- mmdet
  |-- tools
  |-- workspace
  |  |-- configs    各実験の設定ファイル群
  |  |-- scripts    実験実行用script
  |  |-- work_dirs  実験生成物(pthなど，実行時自動生成)
~~~

## 環境構築 & 実行方法

現状，docker上で実行する方法のみに対応しています．

1. repogitoryをcloneし，ブランチをcontrastiveに変更
    ~~~
    git clone https://github.com/namihagi/mmdetection.git
    cd mmdetection
    git checkout contrastive
    ~~~
2. datasetを上記のディレクトリ構造通りに配置
3. docker image をbuild
    ~~~
    cd docker
    docker build . -t mmdetection
    ~~~
4. docker container を起動
    ~~~
    docker run -it --gpus all --shm-size 16gb \
      -v /path/to/mmdetection:/mmdetection \
      mmdetection /bin/bash
    ~~~
5. scriptsを実行
    ~~~
    # in a container
    cd /mmdetection/workspace/scripts
    [shell を実行]
    ~~~
