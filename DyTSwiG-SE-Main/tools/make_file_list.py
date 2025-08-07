import os


def generate_wav_list(source_dir, target_dir, output_file):
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"源目录 {source_dir} 不存在")
        return

    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历源目录中的所有文件
        for filename in os.listdir(source_dir):
            # 检查文件是否为 .wav 文件
            if filename.endswith('.wav'):
                # 获取无后缀的文件名
                name_without_ext = os.path.splitext(filename)[0]
                # 创建目标目录下的完整文件路径
                full_target_path = os.path.join(target_dir, filename)
                # 写入一行到txt文件，格式为：无后缀文件名|目标目录+文件名.wav
                f.write(f"{name_without_ext}|{full_target_path}\n")


# 使用示例
source_directory = '/home/xyj/uyghur/clean_wavs/'  # 替换为你的源目录路径
target_directory = '/home/xyj/uyghur/clean_wavs'  # 替换为目标目录路径
output_filename = 'uyghur_train.txt'  # 替换为你想要的输出文件名

generate_wav_list(source_directory, target_directory, output_filename)
