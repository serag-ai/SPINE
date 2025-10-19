rm -rf ./M3D/LaMed/script/finetune_lora.sh; cp ./src/finetune_lora.sh ./M3D/LaMed/script/
rm -rf ./M3D/LaMed/src/dataset/multi_dataset.py; cp ./src/multi_dataset.py ./M3D/LaMed/src/dataset/
rm -rf ./M3D/LaMed/src/dataset/prompt_templates.py; cp ./src/prompt_templates.py ./M3D/LaMed/src/dataset/

cp ./src/merge_lora_weights_and_save_hf_model.py ./M3D/merge_lora_weights_and_save_hf_model.py
cp ./src/demo_csv.py ./M3D/demo_csv.py
cp ./src/train.py ./M3D/custom_train.py
