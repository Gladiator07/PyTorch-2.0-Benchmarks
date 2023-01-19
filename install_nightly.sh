pip install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117
pip install git+https://github.com/huggingface/transformers.git@main --upgrade
pip install datasets evaluate wandb omegaconf