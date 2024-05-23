# import re

# with open('./NeuroGauss4D/log/4DGS_Exper_macpool_v3/wo_mlp/NL_test2.txt', 'r', encoding="utf-8") as file:
#     text = file.read()

# pattern = r'Final Result CD Loss is:(\d+\.\d+)|\
# Final Result EMD Loss is:(\d+\.\d+)|\
# Final Frame-1 Result CD Loss is:(\d+\.\d+)|\
# Final Frame-1 Result EMD Loss is:(\d+\.\d+)|\
# Final Frame-2 Result CD Loss is:(\d+\.\d+)|\
# Final Frame-2 Result EMD Loss is:(\d+\.\d+)|\
# Final Frame-3 Result CD Loss is:(\d+\.\d+)|\
# Final Frame-3 Result EMD Loss is:(\d+\.\d+)'

# matches = re.findall(pattern, text)

# averages = {}
# for match in matches:
#     for i, value in enumerate(match):
#         if value:
#             key = ['Final Result CD Loss', 'Final Result EMD Loss',
#                    'Final Frame-1 Result CD Loss', 'Final Frame-1 Result EMD Loss',
#                    'Final Frame-2 Result CD Loss', 'Final Frame-2 Result EMD Loss',
#                    'Final Frame-3 Result CD Loss', 'Final Frame-3 Result EMD Loss'][i]
#             if key in averages:
#                 averages[key].append(float(value))
#             else:
#                 averages[key] = [float(value)]

# for key, values in averages.items():
#     average = sum(values) / len(values)
#     print(f'{key}: {average}')




################## DHB  ################### 
import pdb
def read_losses_from_file(file_path):
    cd_losses = []
    emd_losses = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Final Result CD Loss is:"):
                cd_loss = float(line.split(':')[1])
                cd_losses.append(cd_loss)
            elif line.startswith("Final Result EMD Loss is:"):
                emd_loss = float(line.split(':')[1])
                emd_losses.append(emd_loss)

    average_cd_loss = sum(cd_losses) / len(cd_losses) if cd_losses else 0
    average_emd_loss = sum(emd_losses) / len(emd_losses) if emd_losses else 0
    
    return average_cd_loss, average_emd_loss

file_path = './NeuroGauss4D/script/txt/G_DHB.txt'

average_cd_loss, average_emd_loss = read_losses_from_file(file_path)
print(f"Average Final Result CD Loss: {average_cd_loss}")
print(f"Average Final Result EMD Loss: {average_emd_loss}")