# CUDA_VISIBLE_DEVICES=0 python inference.py \
#     --config_path configs/self_forcing_dmd.yaml \
#     --output_folder videos/self_forcing_dmd \
#     --checkpoint_path /nvfile-heatstorage/AIGC_H100/cyb/World_DMD_9.4/logs/self_forcing_dmd_32GPU_8000/model.pt \
#     --data_path prompts/MovieGenVideoBench_extended.txt \
#     --use_ema

python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path /nvfile-heatstorage/AIGC_H100/cyb/World_DMD_9.4/logs/self_forcing_dmd_32GPU_8000/model.pt \
    --data_path ./prompts/test_txt.txt \
    --use_ema
