import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset

import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta

from ..utils.utils import mask2box
from .dataset_info import dataset_info
from .prompt_templates import Caption_templates,CapSeg_templates
from .term_dictionary import term_dict
import nibabel as nib
from functools import partial

class SpineCapDataset(Dataset):
        def __init__(self, args, tokenizer, target_shape=(256, 256, 128), mode="train"):
            self.args = args
            self.data_root = args.data_root
            self.tokenizer = tokenizer
            self.mode = mode

            self.image_tokens = "<im_patch>" * args.proj_out_num

            df = pd.read_csv(
                args.amos_train_cap_data_path
                if mode == "train"
                else args.amos_validation_cap_data_path
            )
            self.images_path = df["image_path"]
            self.captions = df["Clinician's Notes"]
            #self.organs = df["label"]

            self.nii_to_tensor = partial(
                self.__nii_img_to_tensor, target_shape=target_shape
            )

        def __nii_img_to_tensor(self, path, target_shape):
            img_data = nib.load(path)
            img_data = img_data.get_fdata()
            img_data = img_data.astype(np.float32)

            img_data = np.transpose(img_data, (1, 2, 0))
            # img_data = img_data * 1000
            # hu_min, hu_max = -1000, 200
            #img_data = np.clip(img_data, hu_min, hu_max)

            #img_data = (((img_data + 400) / 600)).astype(np.float32)
            slices = []
            # Use this part only for m3d
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

            tensor = torch.tensor(img_data)

            # Get the dimensions of the input tensor

            # Extract dimensions
            h, w, d = tensor.shape
            # Calculate cropping/padding values for height, width, and depth
            dh, dw, dd = target_shape
            h_start = max((h - dh) // 2, 0)
            h_end = min(h_start + dh, h)
            w_start = max((w - dw) // 2, 0)
            w_end = min(w_start + dw, w)
            d_start = max((d - dd) // 2, 0)
            d_end = min(d_start + dd, d)

            # Crop or pad the tensor
            tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

            pad_h_before = (dh - tensor.size(0)) // 2
            pad_h_after = dh - tensor.size(0) - pad_h_before

            pad_w_before = (dw - tensor.size(1)) // 2
            pad_w_after = dw - tensor.size(1) - pad_w_before

            pad_d_before = (dd - tensor.size(2)) // 2
            pad_d_after = dd - tensor.size(2) - pad_d_before

            tensor = torch.nn.functional.pad(
                tensor,
                (
                    pad_d_before,
                    pad_d_after,
                    pad_w_before,
                    pad_w_after,
                    pad_h_before,
                    pad_h_after,
                ),
                value=0,
            )

            tensor = tensor.permute(2, 0, 1)
            #print("tensor.shape")

            tensor = tensor.unsqueeze(0).unsqueeze(0)
            return tensor[0]

        def __len__(self):
            return len(self.images_path)

        def __getitem__(self, idx):
            max_attempts = 100
            for _ in range(max_attempts):
                try:
                    image = self.nii_to_tensor(
                        os.path.join(self.data_root, self.images_path[idx])
                    )

                    answer = self.captions[idx]
                    prompt_question = random.choice(Caption_templates)

                    question = self.image_tokens + prompt_question

                    text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )  # <IMG_TOKENS><QUESTION>' '<ANSWER/CAP>

                    input_id = text_tensor["input_ids"][0]
                    attention_mask = text_tensor["attention_mask"][0]

                    valid_len = torch.sum(attention_mask)
                    if valid_len < len(input_id):
                        input_id[valid_len] = self.tokenizer.eos_token_id

                    question_tensor = self.tokenizer(
                        question,
                        max_length=self.args.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )

                    question_len = torch.sum(question_tensor["attention_mask"][0])

                    label = input_id.clone()
                    label[:question_len] = -100
                    if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                        label[label == self.tokenizer.pad_token_id] = -100
                        if valid_len < len(label):
                            label[valid_len] = self.tokenizer.eos_token_id
                    else:
                        label[label == self.tokenizer.pad_token_id] = -100

                    ret = {
                        "image": image,
                        "input_id": input_id,
                        "label": label,
                        "attention_mask": attention_mask,
                        "question": question,
                        "answer": answer,
                        "question_type": "Caption",
                    }
                    return ret

                except Exception as e:
                    print(f"Error in __getitem__ at index {idx}: {e}")
                    idx = random.randint(0, self.__len__() - 1)

class SpineCapSegDataset(Dataset):
    def __init__(self, args, tokenizer, target_shape=(256, 256, 6), mode="train"):
        self.args = args
        self.data_root = args.data_root
        #self.data_root = args.seg_data_path
        self.tokenizer = tokenizer
        self.mode = mode

        #self.data_list = pd.read_csv(args.seg_data_path, engine='python')

        self.image_tokens = "<im_patch>" * args.proj_out_num

        df = pd.read_csv(
            args.refseg_data_train_path 
            if mode == "train" 
            else args.refseg_data_test_path
        )

        # Convert series to numpy arrays to avoid pandas index issues
        self.images_path = df["image_path"].values
        self.captions = df["Clinician's Notes"].values
        self.seg = df["mask_path"].values

        self.nii_to_tensor = partial(self.__nii_img_to_tensor, target_shape=target_shape)

    def __nii_img_to_tensor(self, path, target_shape, is_seg=False):
        img_data = nib.load(path).get_fdata()

        if is_seg:
            # label_map = {50: 0, 100: 1, 150: 2, 200: 3, 250: 4}
            # # for k, v in label_map.items():
            # #     img_data[img_data == k] = v

            # # img_data = img_data.astype(np.int64)
            # img_data = np.vectorize(label_map.get)(img_data.astype(np.int32))
            # img_data[np.isnan(img_data)] = 250  # or your background class
            # img_data = img_data.astype(np.int64)
            img_data = img_data.astype(np.int32)
            img_data = np.where(img_data == 250, 0, 1).astype(np.uint8)
        else:
            img_data = img_data.astype(np.float32)
            if self.mode == "train":
                # Normalize only during training
                img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
                    
        img_data = np.transpose(img_data, (1, 2, 0))    
        
        tensor = torch.tensor(img_data)

        # Pad/crop to match the target shape
        h, w, d = tensor.shape
        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(
            tensor,
            (
                pad_d_before,
                pad_d_after,
                pad_w_before,
                pad_w_after,
                pad_h_before,
                pad_h_after,
            ),
            value=0,
        )

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor[0]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        try:
            image_path = os.path.join(self.data_root, self.images_path[idx])
            seg_path = os.path.join(self.data_root, self.seg[idx])

            image = self.nii_to_tensor(image_path, is_seg=False)
            mask = self.nii_to_tensor(seg_path, is_seg=True)
            #print("Mask info", mask)

           
            prompt_question = random.choice(Caption_templates)
            seg_question = random.choice(CapSeg_templates)
            question = self.image_tokens +' ' +prompt_question + seg_question

            answer = self.captions[idx]

            self.tokenizer.padding_side = "right"
            text_tensor = self.tokenizer(
                question + " " + answer,
                max_length=self.args.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            question_tensor = self.tokenizer(
                question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            label = input_id.clone()
            label[:question_len] = -100
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                label[label == self.tokenizer.pad_token_id] = -100
                if valid_len < len(label):
                    label[valid_len] = self.tokenizer.eos_token_id
            else:
                label[label == self.tokenizer.pad_token_id] = -100

            return {
                "image": image,
                "input_id": input_id,
                "label": label,
                "seg": mask,
                "attention_mask": attention_mask,
                "question": question,
                "answer": answer,
                "question_type": "refseg",
            }

        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            idx = random.randint(0, self.__len__() - 1)

# class MultiSegDataset(Dataset):
#     def __init__(self, args, tokenizer, mode='train'):
#         super(MultiSegDataset, self).__init__()
#         self.tokenizer = tokenizer

#         self.dataset_info = dataset_info

#         self.ds_list = []
#         # self.ds_list.append(RefSegDataset(args, tokenizer, mode=mode))
#         for dataset_code in self.dataset_info.keys():
#             self.ds_list.append(SpineCapSegDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
#             self.ds_list.append(SpineCapSegDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
#         self.dataset = ConcatDataset(self.ds_list)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]          
                    
# class SpineCapSegDataset(Dataset):
#     def __init__(self, args, tokenizer,target_shape=(256, 256, 6), mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.seg_data_path=args.seg_data_path
#         self.tokenizer = tokenizer
#         self.mode = mode

#         self.image_tokens = "<im_patch>" * args.proj_out_num

#         df = pd.read_csv(
#                 args.refseg_data_train_path
#                 if mode == "train"
#                 else args.refseg_data_test_path
#             )
        
#         self.images_path = df["image_path"]
#         self.captions = df["Clinician's Notes"]
#         self.seg =df["mask_path"]

#         self.nii_to_tensor = partial(
#                 self.__nii_img_to_tensor, target_shape=target_shape
#             )

#     def __nii_img_to_tensor(self, path, target_shape, is_seg=False):
#         img_data = nib.load(path)
#         img_data = img_data.get_fdata()
        
#         if is_seg: 
#             img_data = img_data.astype(np.int8)
#             img_data = np.transpose(img_data, (1, 2, 0))
#         else:
#             img_data = img_data.astype(np.float32)

#             img_data = np.transpose(img_data, (1, 2, 0))
#             # img_data = img_data * 1000
#             # hu_min, hu_max = -1000, 200
#             #img_data = np.clip(img_data, hu_min, hu_max)

#             #img_data = (((img_data + 400) / 600)).astype(np.float32)
#             slices = []
#             # Use this part only for m3d
#             img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

#             tensor = torch.tensor(img_data)

#             # Get the dimensions of the input tensor

#             # Extract dimensions
#             h, w, d = tensor.shape
#             # Calculate cropping/padding values for height, width, and depth
#             dh, dw, dd = target_shape
#             h_start = max((h - dh) // 2, 0)
#             h_end = min(h_start + dh, h)
#             w_start = max((w - dw) // 2, 0)
#             w_end = min(w_start + dw, w)
#             d_start = max((d - dd) // 2, 0)
#             d_end = min(d_start + dd, d)

#             # Crop or pad the tensor
#             tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

#             pad_h_before = (dh - tensor.size(0)) // 2
#             pad_h_after = dh - tensor.size(0) - pad_h_before

#             pad_w_before = (dw - tensor.size(1)) // 2
#             pad_w_after = dw - tensor.size(1) - pad_w_before

#             pad_d_before = (dd - tensor.size(2)) // 2
#             pad_d_after = dd - tensor.size(2) - pad_d_before

#             tensor = torch.nn.functional.pad(
#                 tensor,
#                 (
#                     pad_d_before,
#                     pad_d_after,
#                     pad_w_before,
#                     pad_w_after,
#                     pad_h_before,
#                     pad_h_after,
#                 ),
#                 value=0,
#             )

#             tensor = tensor.permute(2, 0, 1)
#             #print("tensor.shape")

#             tensor = tensor.unsqueeze(0).unsqueeze(0)
#             return tensor[0]


#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             try:
        
#                 image_path = os.path.join(self.data_root, self.images_path[idx])
#                 seg_path = os.path.join(self.seg_data_path, self.seg[idx])

#                 image = self.nii_to_tensor(image_path, is_seg=False)
#                 mask = self.nii_to_tensor(seg_path, is_seg=True)

#                 # image = self.nii_to_tensor(
#                 #         os.path.join(self.data_root, self.images_path[idx])
#                 #     )
#                 # mask = self.nii_to_tensor(
#                 #         os.path.join(self.data_root, self.images_path[idx])
#                 #     )
#                 answer = self.captions[idx]
#                 prompt_question = random.choice(Caption_templates)
#                 seg_question = random.choice(CapSeg_templates)

#                 question = self.image_tokens + prompt_question + seg_question
#                 self.tokenizer.padding_side = "right"

#                 text_tensor = self.tokenizer(
#                 question + " " + answer,
#                 max_length=self.args.max_length,
#                 truncation=True,
#                 padding="max_length",
#                 return_tensors="pt",)
                
#                 # image_path = os.path.join(self.args.data_root, data["Image"])

#                 # image_array = np.load(image_path)  # 1*32*256*256, normalized

#                 # seg_path = os.path.join(self.args.data_root, data["Mask"])
#                 # seg_array = np.load(seg_path)
#                 # seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

#                 # item = {
#                 #     "image": image,
#                 #     "seg": mask,
#                 # }


#                 # question = data["Question"]
#                 # question = self.image_tokens + ' ' + question

#                 # answer = data["Answer"]

#                 # self.tokenizer.padding_side = "right"
#                 # text_tensor = self.tokenizer(
#                 #     question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
#                 # )

#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]

#                 valid_len = torch.sum(attention_mask)
#                 if valid_len < len(input_id):
#                     input_id[valid_len] = self.tokenizer.eos_token_id

#                 question_tensor = self.tokenizer(
#                     question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
#                 )
#                 question_len = torch.sum(question_tensor["attention_mask"][0])

#                 label = input_id.clone()
#                 label[:question_len] = -100
#                 if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
#                     label[label == self.tokenizer.pad_token_id] = -100
#                     if valid_len < len(label):
#                         label[valid_len] = self.tokenizer.eos_token_id
#                 else:
#                     label[label == self.tokenizer.pad_token_id] = -100

#                 ret = {
#                     'image': image,
#                     'input_id': input_id,
#                     'label': label,
#                     'seg': mask,
#                     'attention_mask': attention_mask,
#                     'question': question,
#                     'answer': answer,
#                     'question_type': "refseg",
#                 }

#                 return ret

#             except Exception as e:
#                 print(f"Error in __getitem__ at index {idx}: {e}")
#                 idx = random.randint(0, len(self.data_list) - 1)