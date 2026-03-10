import pyarrow as pa
import pyarrow.parquet as pq

def txt_to_parquet(txt_path, pq_path, separator="<|endoftext|>"):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    stories = [s.strip() for s in text.split(separator) if s.strip()]
    
    table = pa.table({"text": stories})
    
    pq.write_table(
        table,
        pq_path,
        compression="zstd",  # 这里可以正常用
        use_dictionary=True,
        write_statistics=True,
    )
    
    print(f"Converted {len(stories)} stories to {pq_path}")

# 使用
txt_to_parquet(
    "datasets/TinyStories-train.txt",
    "datasets/tinystories_train.parquet"
)
