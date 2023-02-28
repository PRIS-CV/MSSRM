# MSSRM
An implementation of MSSRM method

# Environment 
python >=3.7<br />
pytorch >=1.7<br />
opencv-python >=4.0<br />
scipy >=1.4.0<br />
h5py >=2.10<br />
pillow >=7.0.0<br />

# Datasets
 Download SR-Crowd dataset from [Baidu-Disk](https://pan.baidu.com/s/1NkJMKIw6iSn2_ssbIaecvQ?pwd=mvi3), passward:mvi3 ; or [Google-Drive](https://drive.google.com/drive/folders/1XaYH8KPxMtK6j6_LkLG_EJGaZ1j9qa4m?usp=share_link)

# Model
Download the pretrained model from from [Baidu-Disk](https://pan.baidu.com/s/1Be09E7mn9Y0EHv4YNKM6Ng?pwd=ma5g ), passward:ma5g ; or [Google-Drive](https://drive.google.com/drive/folders/1NsCkX_aoOV68Bd95Eqo4qP-DHoKa-OxC?usp=share_link)

# Quickly test
* ```git clone https://github.com/Xiejiahao233/MSSRM.git```<br />
  ```cd MSSRM```<br />
* Download Dataset and Model<br />
* Generate images list

  Edit "make_npydata.py" to change the path to your original dataset folder.<br />
  Run ```python make_npydata.py```  .<br />
* Test
  ```python val.py  --test_dataset Crowdsr  --pre ./model/Crowdsr/model_best.pth --gpu_id 0```<br />

# Training
  for x2 upscale:<br />
    ```python train.py --upscale x2 --gpu_id 0```<br />
  for x4 upscale:<br />
    ```python train.py --upscale x4 --gpu_id 0```<br />
