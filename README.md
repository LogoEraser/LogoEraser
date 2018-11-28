# LogoEraser
AI스쿨 인공지능 R&amp;D 실무자 양성과정 2기 6조 프로젝트

# Keras RetinaNet Logo Detection
Keras implementation of RetinaNet object detection on logo detection. Forked on https://github.com/fizyr/keras-retinanet. Original paper is [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

## Preparation(진행하다가 모듈 없다는 오류 메세지 나오면 설치하시면서 진행하시면 됩니다.)

1) Clone this repository.
2) In the repository `keras-retinanet` execute `python setup.py install --user`.
   Please make sure `tensorflow` is installed as per your systems requirements.
   Also, make sure Keras 2.1.2 is installed.
3) This repository requires the master branch of `keras-resnet` (run `pip install --user --upgrade git+https://github.com/broadinstitute/keras-resnet`).


## Training(4번 과정 생략해주세요, 데이터는 프로젝트 root directory에)

1) Make sure to complete Preparation steps first.
2) Download FlickrLogos32 dataset from [here](http://www.multimedia-computing.de/flickrlogos/data/FlickrLogos-32_dataset_v2.zip).
   Extract `FlickrLogos-v2` folder.
3) Download Logos-32plus_v1.0.1.zip dataset from [here](https://goo.gl/FPtqxR).
   Extract `images` folder and `groundtruth.mat` file to `Logos32plus` folder.
4) In main repository run command `python prepare.py -f ./../FlickrLogos-v2/ -l ./../Flickr32plus/ -c ./csvpaths/classes.csv -t ./csvpaths/retina-train.csv -v ./csvpaths/retina-valid.csv -s ./csvpaths/retina-test.csv`.
   Make sure `-f` option is FlickrLogos dataset folder path, `-l` is Logos32plus dataset path.
   In csvpaths folder files `retina-valid.csv`, `retina-train.csv` and `retina-test.csv` should have appeared.
5) Now run `python train.py -n name_of_snapshot_folder -c ./csvpaths/classes.csv -t ./csvpaths/retina-train.csv -v ./csvpaths/retina-valid.csv`.
   Folder with weighs and metadata should appear in `snapshots` folder.


## Evaluating classification model(안하셔도 돼요)

1) Make sure to complete Preparation steps first.
   Make sure to do 2-4 Training steps. 
2) Train your own model or download weights from [here](https://drive.google.com/file/d/1eDybynuRoSvTXPMRqjeHLFliI42ZP4tx/view?usp=sharing).
3) In this repository run command `python evaluate.py -w weights.h5 -c ./csvpaths/classes.csv -t ./csvpaths/retina-test.csv -o ./evalkit/classification.txt`.
   `-w` is path to weights and `-o` is output path.
   In `evalkit` folder `classification.txt` file should appear.
4) In `evalkit` folder run command `python fl_eval_classification.py --flickrlogos=..\..\FlickrLogos-v2 --classification="classification.txt"`.
   Make sure `--flickrlogos` option is path to FlickLogos32 dataset and `--classification` option is txt file from step 3.
   You can use `original-classification.txt` which is made on default weights.


## Evaluating single photo or video(2번 과정은 제가 학습시킨 모델 다운로드 링크로 수정했습니다. 4번 스크립트는 제가 조금 수정했습니다. -e erase/no: 로고 지우기/ detection만)

1) Make sure to complete Preparation steps first.
2) Train your own model or download weights from [here](https://drive.google.com/open?id=1-DyNWV7jUybNpk0zoNdh8bxeJV0XoK9x).
3) To evaluate photo run command `python test.py -f ./examples/test.png -o ./examples/output.png -w weights.h5 -c ./csvpaths/classes.csv`.
   Where `-f` is your photo, `-o` is output photo, `-w` is weights.
4) To evaluate video run command `python test_video.py -f ./examples/video.mp4 -o ./examples/output_video.mp4 -w weights.h5 -c ./csvpaths/classes.csv -e erase`.


## Dependencies

1) Tensorflow (https://www.tensorflow.org/install/).
2) Keras 2.1.2 `pip install keras` install after Tensorflow.
3) OpenCV `pip install opencv-python`.
