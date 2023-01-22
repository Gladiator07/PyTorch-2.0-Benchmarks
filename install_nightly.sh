pip3 install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
pip install git+https://github.com/huggingface/transformers.git@main --upgrade
pip install datasets evaluate wandb omegaconf sentencepiece
pip install git+https://github.com/huggingface/accelerate@main