wget -P datasets/ https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
uv run utils/convert_txt_to_pq.py