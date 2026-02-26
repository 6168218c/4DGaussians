from torch.utils.data import Dataset
from scene.cameras import Camera, SimpleCamera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
    
                mask = caminfo.mask
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
    
def compute_override_width_heights(override_h, override_w, image_height, image_width):
    if override_h == -1 and override_w == -1:
        override_w = 800
    if override_h == -1:
        override_h = int(image_height / image_width * override_w)
    elif override_w == -1:
        override_w = int(image_width / image_height * override_h)
    # round to multiples of 32
    override_w = ((override_w - 1) // 32 + 1) * 32
    override_h = ((override_h - 1) // 32 + 1) * 32
    return override_h, override_w

class FourDGSEditDataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        override_h, override_w = compute_override_width_heights(args.override_h, args.override_w, dataset.image_height, dataset.image_width)
        self.override_h = override_h
        self.override_w = override_w
        print(f"Setting Override height: {self.override_h}, Override width: {self.override_w}")
        self.dataset_type=dataset_type
        self.edited_original_images = {}
        
    def set_image(self, idxs, image):
        for idx, img in zip(idxs, image):
            self.edited_original_images[idx] = img
        
    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX, FovY = self.dataset.get_override_fov(self.override_h / self.override_w)
                mask=None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                assert False, "FoV conversions not implemented yet"
                time = caminfo.time
    
                mask = caminfo.mask
            return SimpleCamera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,
                                image_width=self.override_w,image_height=self.override_h,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask, image=self.edited_original_images.get(index, None))
        else:
            return self.dataset[index]
    def __len__(self):
        return len(self.dataset)