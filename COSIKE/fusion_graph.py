import torch
import torch.nn as nn
import pickle

class FusionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(FusionModel, self).__init__()
        # 线性层
        self.linear_layer = nn.Linear(input_size * 4, output_size)

    def forward(self, node_representations):
        # 将 9 种节点表示拼接起来
        fusion_representations = torch.cat(node_representations, dim=1)
        # 通过线性层获得最终的融合后的表示
        final_representation = self.linear_layer(fusion_representations)
        return final_representation


# 创建模型实例
input_size = 2048  # 输入特征的维度
output_size = 2048 # 输出特征的维度


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# 创建模型实例，并将其移动到设备上
model = FusionModel(input_size, output_size).to(device)

node_representations = []  # 在循环外部定义空列表

for i in range(1, 10):
    if i!=3 and i!=4 and i!=7 and i!=8 and i!=9:
        with open(f'/home/stu4/project/lightgcn/data/meld_word/{i}/result/meld_word/layer1/word_embedding_latter.pkl', 'rb') as file:
            representation = pickle.load(file)
            representation_tensor = torch.tensor(representation).to(device)
            node_representations.append(representation_tensor)

# 将所有张量移动到设备上
node_representations = [tensor.to(device) for tensor in node_representations]

# 将列表中的张量传递给模型
output = model(node_representations)


# 指定要保存的文件名
output_file = '/home/stu4/project/meld-ccac/graph2/jubu+quanju/quanju_after_ronghe_2048.pkl'

# 将output保存到.pkl文件中
with open(output_file, 'wb') as file:
    pickle.dump(output, file)

print(output)
print(output.size())
print(len(output))




# 使用 concatenate 函数将列表中的数组沿着第一个轴连接
            #concatenated_array = np.concatenate(all_logits, axis=0)

            # 将结果展平成（12，2048）形状的数组
            #reshaped_array = concatenated_array.reshape(-1, concatenated_array.shape[-1])
            # 打开文件并将数组保存为.pkl文件
            #with open("/home/stu4/project/CCAC2023/save/val_text_robert.pkl", "wb") as file:
            #    pickle.dump(reshaped_array, file)

            #print("Array saved to text_robert.pkl")
            #print(reshaped_array.shape)  # 输出 (12, 2048)

# -*- coding: gbk -*-
'''import pickle


with open('/home/stu4/project/CCAC2023/graph/kn/knowedge_embedding_latter.pkl', 'rb') as f:
    kn_word_vectors_tensor = pickle.load(f)
    print(kn_word_vectors_tensor)'''