import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from LaMed.src.dataset.multi_dataset import SpineCapDataset,SpineCapSegDataset
from Bench.eval.metrics import BinaryDice
import SimpleITK as sitk
from LaMed.src.model.language_model import LamedPhi3ForCausalLM


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/Spinal_cord/src/results_phi3/automation_cap_amos_short_ctrate_filtered/lr5e5/merged/5",
        #choices=[],
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument(
        "--data_root",
        type=str,
        default="/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/Spinal_cord/src/test",
    )
    # caption data
    parser.add_argument("--amos_train_cap_data_path", type=str, default="./Data/data")
    parser.add_argument(
        "--amos_validation_cap_data_path",
        type=str,
        default="/acfs-home/hoh4002/serag_AI_lab/users/hoh4002/eICU/Spinal_cord/src/test_split.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
    )
    parser.add_argument('--vis', type=bool, default=True)
    parser.add_argument('--seg_enable', type=bool, default=True)

    parser.add_argument("--proj_out_num", type=int, default=256)

    return parser.parse_args(args)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path, device_map="auto", trust_remote_code=True
    # )

    model = LamedPhi3ForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", cache_dir=None
    )

    ckpt = torch.load(
        os.path.join(args.model_name_or_path, "merged_model.bin"),
        map_location="cpu",
    )
    model.load_state_dict(ckpt, strict=True)
    print("load pretrained MLLM weights.")

    model = model.to(device=device)

    model.eval()
    test_dataset = SpineCapDataset(
        args,
        tokenizer,
        target_shape=(
            256,
            256,
            32,
        ),
        mode="validation",
    )  # test1k

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=32,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    metric_fn = BinaryDice()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, "eval_caption.csv")

    with open(output_path, mode="w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question", "Ground Truth", "pred", "Dice"])
        with torch.no_grad():
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                answer = sample["answer"]
    

                input_id = tokenizer(question, return_tensors="pt")["input_ids"].to(
                    device=device
                )
                image = sample["image"].to(device=device)
                # seg = sample["seg"].to(device=device)

                #seg = seg * (original_max - original_min) + original_min

                # with torch.inference_mode():
                #     generation, logits = model.generate(
                #         image, 
                #         input_id, 
                #         seg_enable=args.seg_enable, 
                #         max_new_tokens=args.max_new_tokens, 
                #         do_sample=args.do_sample, 
                #         top_p=args.top_p, 
                #         temperature=args.temperature
                #     )

                # generated_texts = tokenizer.batch_decode(
                #     generation, skip_special_tokens=True
                # )

                # pred = (torch.sigmoid(logits) > 0.5) * 1.0
                # dice = metric_fn(logits, seg).item()


                generation = model.generate(
                    image,
                    input_id,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                generated_texts = tokenizer.batch_decode(
                    generation, skip_special_tokens=True
                )

                result = dict()
                print(f"ANSWER: {answer}")
                print(f"PREDICTON:{generated_texts}")
                # print(f"DICE:{dice}")
                writer.writerow(
                    [
                        question[0],
                        answer[0],
                        generated_texts[0]
                    ]
                )

                # if args.vis:
                #     desired_length = 4

                #     path = os.path.join(args.output_dir, 'eval_seg', str(id).zfill(desired_length))
                #     folder = os.path.exists(path)
                #     if not folder:
                #         os.makedirs(path)

                #     batch_size, z = image.shape[0], image.shape[2]
                #     for b in range(batch_size):
                #         with open(path + '/text.txt', 'a') as f:
                #             f.write("Question: " + question[b] + "\n")
                #             f.write("Answer: " + answer[b] + "\n")
                #             f.write("pred: " + generated_texts[b] + "\n")

                #         out = sitk.GetImageFromArray(image[b][0].detach().cpu().numpy())
                #         sitk.WriteImage(out, os.path.join(path, 'image.nii.gz'))

                #         out = sitk.GetImageFromArray(seg[b][0].detach().cpu().numpy())
                #         sitk.WriteImage(out, os.path.join(path, 'seg.nii.gz'))

                #         out = sitk.GetImageFromArray(pred[b][0].detach().cpu().numpy())
                #         sitk.WriteImage(out, os.path.join(path, 'pred.nii.gz'))


if __name__ == "__main__":
    main()
