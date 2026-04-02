import os
import csv

def merge_csv_files(input_dir, output_file):
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and not f == output_file]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    # 读取第一个文件的表头
    first_file = os.path.join(input_dir, csv_files[0])
    with open(first_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    # 写入合并后的文件
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        
        # 遍历所有CSV文件
        for csv_file in csv_files:
            file_path = os.path.join(input_dir, csv_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    # 跳过表头
                    next(reader, None)
                    # 写入数据行
                    for row in reader:
                        if row:
                            writer.writerow(row)
                print(f"Merged: {csv_file}")
            except Exception as e:
                print(f"Error merging {csv_file}: {e}")
    
    print(f"\nAll files merged into: {output_file}")

if __name__ == "__main__":
    input_directory = "result"
    output_file = "result/merged_all.csv"
    merge_csv_files(input_directory, output_file)