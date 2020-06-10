----------------------
Nhận diện từ vựng tiếng Anh
----------------------
Hệ thống nhận dạng từ vựng tiếng Anh với đầu vào là một âm thanh mới của một trong các từ có trong CSDL.

## Data

Bộ data gồm 10 từ (swing, brick, high, book, minimal, circle, international, agreement, labor, ordinary), mỗi từ gồm nhiều loại giọng nói khác nhau. Các file âm thanh được lưu trữ dưới dạng wav. 
 Xem chi tiết data [train](https://drive.google.com/drive/folders/1w3Cpyy8_F16e8pIqbmbpztt7o7mYikJk?usp=sharing), [test](https://drive.google.com/drive/folders/1zHSIifQkHrwnIAYNknnhwry_ZGF_YYMr?usp=sharing)

## Cài đặt 

Cài đặt library add-in file [Requirements](requirements.txt). Để install chạy:

```bash
    python -m pip install --upgrade pip && pip3 install -r requirements.txt
```

Cây thư mục như sau:


    root
    ├── v01
    ├── v02
    │   ├── feature
    │   ├── indextree
    │   ├── EnglishSoundSearch.py
    │   ├── featureExtraction.py
    │   ├── SearchTree.py
    │   └── Utils.py
    └──  requirements.txt
    ├── test
    └── train

