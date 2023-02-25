# MSSRM
An implementation of MSSRM method

# Environment 
python >=3.7
pytorch >=1.7
opencv-python >=4.0
scipy >=1.4.0
h5py >=2.10
pillow >=7.0.0

# Datasets
Coming soon

# Model
Download the pretrained model from

# Quickly test
· git clone https://github.com/Xiejiahao233/MSSRM.git
  cd MSSRM
· Download Dataset and Model
· Generate images list

  Edit "make_npydata.py" to change the path to your original dataset folder.
  Run python make_npydata.py  .
· Test
  python val.py  --test_dataset Crowdsr  --pre ./model/Crowdsr/model_best.pth --gpu_id 0

# Training
  for x2 upscale:
    python train.py --upscale x2 --gpu_id 0
  for x4 upscale
    python train.py --upscale x4 --gpu_id 0
