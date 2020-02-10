 -*- coding: utf8 -*-

HO CHI MINH CITY UNIVERSITY OF TECHNOLOGY
       NGUYEN THANH DAT - 1510698
LVTN: Nhận dạng ngôn ngữ ký hiệu cho người khiếm thính sử dụng kỹ thuật học sâu:
Tách và phân tích đặc trưng khung xương trên video RGB

guthub: https://github.com/Dreamer179
email: thanhdatbku97@gmail.com


Chương trình chạy trên hệ điều hành ubuntu 16.x, 18.x

B1: Cài các thư viện cần thiết
$ pip3 install -r requirements.txt

B2: Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

Sau khi cài các thư viện và build C++ library thành c
B3:Collect data
$ python collect_data.py

B4: Mở file "train_16_main.ipynb" bằng jupyter-notebook để train và đánh giá mô hình.

B5: Bỏ file đã train vào folder "model" và tiến hành chạy nhận diện
$ python main.py

Lưu ý: 
- Có thể chọn lựa các model ước tình khung xương trong folder "models"
- Chương trình sẽ thực thi nhanh hơn khi sử dụng tensorflow-gpu
