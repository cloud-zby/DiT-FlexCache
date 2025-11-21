set -x
#  cd $sbp_dir
#  sudo docker run -v/nvfile-heatstorage/:/nvfile-heatstorage/ -v $(pwd):/workspace -it --gpus all torch-2.5.0 bash misc/241105_sdxl_example/run.sh

script_dir=$(dirname "$0" | xargs -n1 realpath)
parent_dir=$(dirname "$script_dir")
testsets=${1:-"${parent_dir}/benchmarks/example_msbp.jsonl"}
output_dir=${2:-"${parent_dir}/output"}
lora_path="/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/ckpt-step00028000.safetensors"
shift; shift

SDXL_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/aeab11b103b9211559139b4039011f634722a758
VAE_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/sdxl-vae-fp16-fix
PERSON_DET_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/yolov8x-seg.pt
IMAGE_ENCODER_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/urm-l-008f.pth

FLUX_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/FLUX.1-dev
CONDITIONER_BASE_MODEL_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/eva02_large_patch14_448.mim_in22k/model.safetensors
FLUX_ADAPTER_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/flux_adapter_step-180000.safetensors
# MSBP_MODEL_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/DreamO/dreamo.safetensors
MSBP_MODEL_PATH_1=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/DreamO/msbp_num_1.safetensors
MSBP_MODEL_PATH_2=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/DreamO/msbp_num_2_3.safetensors
BEN2_MODEL_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/BEN2/BEN2_Base.pth
FLUX_FILL_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/FLUX.1-Fill-dev
FLUX_FILL_ADAPTER_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/flux_fill_adapter_step-060000.safetensors
FLUX_CONTROLNET_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/flux_controlnet 

QWENVL_MODEL_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/Qwen2.5-VL-32B-Instruct

FACE_MODEL_PATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/insightface
 
export PYTHONPATH=/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/id-reverse-diffusion/

python $script_dir/infer7_with_pose.py \
    --lora_path $lora_path \
    --testsets $testsets \
    --output_dir ${output_dir}/infer7_with_pose/test_0828 \
    --sdxl_path $SDXL_PATH \
    --vae_path $VAE_PATH \
    --flux_path $FLUX_PATH \
    --conditioner_base_model_path $CONDITIONER_BASE_MODEL_PATH \
    --flux_ip_model_path $FLUX_ADAPTER_PATH \
    --msbp_model_path_1 ${MSBP_MODEL_PATH_1} \
    --msbp_model_path_2 ${MSBP_MODEL_PATH_2} \
    --ben2_model_path ${BEN2_MODEL_PATH} \
    --flux_fill_model_path $FLUX_FILL_PATH \
    --flux_ip_fill_model_path $FLUX_FILL_ADAPTER_PATH \
    --qwenvl_model_path $QWENVL_MODEL_PATH \
    --flux_controlnet_model_path $FLUX_CONTROLNET_PATH \
    --person_detection_path $PERSON_DET_PATH \
    --image_encoder_path $IMAGE_ENCODER_PATH \
    --face_model_path $FACE_MODEL_PATH $@
