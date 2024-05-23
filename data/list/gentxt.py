
# file_names_and_content = [
#     "longdress",
#     "loot",
#     "redandblack",
#     "soldier",
#     "squat_2_fps1024_aligned",
#     "swing_fps1024_aligned"
# ]

# # 遍历列表，为每个文件名和内容创建一个文本文件
# for name in file_names_and_content:
#     # 使用with语句确保文件正确关闭
#     with open(f"{name}.txt", 'w') as file:
#         file.write(name)

# print("文件已生成。")

# 打开原始文件并读取所有行
import random
num_spilt = 16
with open('./NeuroGauss4Ddata/NL_Drive/test/scene_list/scene_list.txt', 'r') as file:
    lines = file.readlines()

random.shuffle(lines)
# 计算每个子文件应有的行数
total_lines = len(lines)
lines_per_file = total_lines // num_spilt
extra_lines = total_lines % num_spilt

# 创建并写入4个子文件
for i in range(num_spilt):
    # 计算每个子文件的起始和结束索引
    start_index = i * lines_per_file
    end_index = start_index + lines_per_file
    if i < extra_lines:  # 如果有额外的行，分配给前面的子文件
        end_index += 1
    
    # 生成子文件的名称
    subfile_name = f'./NeuroGauss4Ddata/NL_Drive_list16/NL_Drive_test_{i}.txt'
    # 写入子文件
    with open(subfile_name, 'w') as subfile:
        subfile.writelines(lines[start_index:end_index])

print("succesd!!!!!!!!!!")