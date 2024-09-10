# !/bin/bash

## Install stuff we need
add-apt-repository -y ppa:rmescandon/yq
apt-get update
apt-get install -y screen vim git-lfs unzip yq

# Debug
env

## INSTALL ai-toolkit and other stuff
cd /workspace
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
pip install accelerate transformers diffusers huggingface_hub torchvision safetensors lycoris-lora==1.8.3 flatten_json pyyaml oyaml tensorboard kornia invisible-watermark einops toml albumentations pydantic omegaconf k-diffusion open_clip_torch timm prodigyopt controlnet_aux==0.0.7 python-dotenv bitsandbytes hf_transfer lpips pytorch_fid optimum-quanto sentencepiece 

## LOGIN HF
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

## SETUP ai-toolkit
cd /workspace/ai-toolkit

# Download images 
wget -O images.zip "${IMAGE_ARCHIVE}"
unzip images.zip -d images

## Write ai-toolkit config with params passed from Colab notebook
wget -O config/ai-toolkit_config.yaml https://raw.githubusercontent.com/geronimi73/RunFlux/main/train-on-my-face/train_lora_flux_24gb.yaml

# non-array vars
export FOLDER_PATH="/workspace/ai-toolkit/images"
export MODEL_NAME="black-forest-labs/FLUX.1-dev"

declare -A yaml_params=(
  [config.process[0].network.linear]=LORA_RANK
  [config.process[0].network.linear]=LORA_ALPHA
  [config.process[0].trigger_word]=TRIGGER_WORD
  [config.process[0].save.save_every]=SAVE_STEPS
  [config.process[0].datasets[0].folder_path]=FOLDER_PATH
  [config.process[0].train.batch_size]=BATCH_SIZE
  [config.process[0].train.steps]=STEPS
  [config.process[0].train.lr]=LEARNING_RATE
  [config.process[0].sample.seed]=SEED
  [config.process[0].sample.sample_every]=SAMPLE_STEPS
  [config.process[0].model.quantize]=QUANTIZE_MODEL
  [config.process[0].model.name_or_path]=MODEL_NAME
)

for param in "${!yaml_params[@]}"; do
  yq eval -i ".${param} = env(${yaml_params[$param]})" config/ai-toolkit_config.yaml
done

# array vars
IFS=';' read -ra SAMPLE_PROMPTS_ARRAY <<< "$SAMPLE_PROMPTS"

for prompt in "${SAMPLE_PROMPTS_ARRAY[@]}"; do
  echo PROMPT: $prompt
  yq eval -i ".config.process[0].sample.prompts += \"$prompt\"" config/ai-toolkit_config.yaml 
done

yq eval -i 'del(.config.process[0].sample.prompts[0])' config/ai-toolkit_config.yaml


# upload config
huggingface-cli upload $HF_REPO config/ai-toolkit_config.yaml

# sleep 30000

## SCHEDULE UPLOADS of samples/adapters every 3 mins 
mkdir -p output/my_first_flux_lora_v1/samples
touch ai-toolkit.log

huggingface-cli upload $HF_REPO output/my_first_flux_lora_v1 --include="*.safetensors" --every=3 &
huggingface-cli upload $HF_REPO ai-toolkit.log --every=3 &

# (for some reason --every does not upload with samples/ dir, no error, no idea -> bash loop)
bash -c 'while true; do huggingface-cli upload $HF_REPO output/my_first_flux_lora_v1/samples samples; sleep 180; done' &

## TRAIN
python run.py config/ai-toolkit_config.yaml 2>&1 | tee ai-toolkit.log

## UPLOAD RESULTS one last time
huggingface-cli upload $HF_REPO output/my_first_flux_lora_v1/samples samples
huggingface-cli upload $HF_REPO output/my_first_flux_lora_v1 --include="*.safetensors"
huggingface-cli upload $HF_REPO ai-toolkit.log

# sleep infinity
sleep 120
runpodctl remove pod $RUNPOD_POD_ID

