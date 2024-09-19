import torch
import torch.nn as nn
from modules.Transformer import  MELDTransEncoder, AdditiveAttention
from modules.CrossmodalTransformer import CrossModalTransformerEncoder
from transformers import RobertaModel
from transformers import BertModel
from modules.SwinTransformer.backbone_def import BackboneFactory
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
import math
warnings.filterwarnings("ignore", category=UserWarning)

'''SwinTransformer'''
class SwinForAffwildClassification(nn.Module):

    def __init__(self, args):
        super(SwinForAffwildClassification,self).__init__()
        self.num_labels = args.num_labels
        self.swin = BackboneFactory(args.backbone_type, args.backbone_conf_file).get_backbone() #加载Swin-tiny模型
        # Classifier head
        self.linear = nn.Linear(512, 64)
        self.nonlinear = nn.ReLU()
        self.classifier = nn.Linear(64, args.num_labels)
        self.tau = args.tau  #temperature parameter

    def forward(self, images_feature=None, is_trg_task=None, labels=None, criterion=None):
        outputs = self.swin(images_feature) 
        outputs = self.linear(outputs)
        outputs = self.nonlinear(outputs)
        logits = self.classifier(outputs) 
        if is_trg_task:
            logits = F.gumbel_softmax(logits, self.tau)  
        if labels is not None:
            loss = criterion(logits, labels) #cross entropy loss
            return loss
        else:
            return logits


'''multimodal model'''
class MultiModalTransformerForClassification(nn.Module):

    def __init__(self, config, kn_word_vectors):
        super(MultiModalTransformerForClassification,self).__init__()
        self.choice_modality = config.choice_modality
        self.num_labels = config.num_labels
        self.get_text_utt_max_lens = config.get_text_utt_max_lens
        self.hidden_size = config.hidden_size
        if config.pretrainedtextmodel_path.split('/')[-1] == 'roberta-large':
            self.text_pretrained_model = 'roberta'
        else:
            self.text_pretrained_model = 'bert'

        self.audio_emb_dim = config.audio_featExtr_dim
        self.audio_utt_Transformernum = config.audio_utt_Transformernum
        self.get_audio_utt_max_lens = config.get_audio_utt_max_lens
        self.kn_word_vectors_tensor = nn.Embedding(num_embeddings=11386, embedding_dim=2048)
        self.kn_word_vectors_tensor.weight.data = kn_word_vectors
        #crossmodal transformer
        self.crossmodal_num_heads_TA = config.crossmodal_num_heads_TA
        self.crossmodal_layers_TA = config.crossmodal_layers_TA
        self.crossmodal_attn_dropout_TA = config.crossmodal_attn_dropout_TA
        
        self.crossmodal_num_heads_TA_V = config.crossmodal_num_heads_TA_V
        self.crossmodal_layers_TA_V = config.crossmodal_layers_TA_V
        self.crossmodal_attn_dropout_TA_V = config.crossmodal_attn_dropout_TA_V
        
        self.vision_emb_dim = config.vision_featExtr_dim + config.num_labels
        self.vision_utt_Transformernum = config.vision_utt_Transformernum
        self.get_vision_utt_max_lens = config.get_vision_utt_max_lens
        
        '''Textual modality through RoBERTa or BERT'''
        if self.text_pretrained_model == 'roberta':
            self.roberta = RobertaModel.from_pretrained(config.pretrainedtextmodel_path)
            self.text_linear = nn.Linear(self.roberta.config.hidden_size, self.hidden_size)
        else:
            self.bert = BertModel.from_pretrained(config.pretrainedtextmodel_path)
            self.text_linear = nn.Linear(self.bert.config.hidden_size, self.hidden_size)

        self.audio_linear = nn.Linear(self.audio_emb_dim, self.hidden_size)
        self.audio_utt_transformer = MELDTransEncoder(config, self.audio_utt_Transformernum,self.get_audio_utt_max_lens, self.hidden_size)  #执行self-attention transformer
    

        self.vision_linear = nn.Linear(512 + self.hidden_size, self.hidden_size)
        self.vision_utt_transformer = MELDTransEncoder(config, self.vision_utt_Transformernum,self.get_vision_utt_max_lens, self.hidden_size)  #执行self-attention transformer

        self.attention = AdditiveAttention(self.hidden_size, self.hidden_size)

        self.CrossModalTrans_TA = CrossModalTransformerEncoder(self.hidden_size, self.crossmodal_num_heads_TA, self.crossmodal_layers_TA, self.crossmodal_attn_dropout_TA)

        self.CrossModalTrans_TA_V = CrossModalTransformerEncoder(self.hidden_size, self.crossmodal_num_heads_TA_V, self.crossmodal_layers_TA_V, self.crossmodal_attn_dropout_TA_V)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        # Define a trainable weight matrix W
        self.W = nn.Parameter(torch.randn(config.num_labels, config.hidden_size))

        self.multi_head_attention_text_gra = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)


        if self.global_or_local_path=='local':
            self.word_vectors = np.load("/home/stu4/project/COSIKE/graph/local/jubu.pkl", allow_pickle=True)
        if self.global_or_local_path=='global':
            self.word_vectors = np.load("/home/stu4/project/COSIKE/graph/global/quanju.pkl", allow_pickle=True)
        if self.global_or_local_path=='local+global':
            self.word_vectors = np.load("/home/stu4/project/COSIKE/graph/local+global/quanju_jubu_2048.pkl", allow_pickle=True)
        
        self.word_vectors_matrix = torch.nn.Embedding(num_embeddings=8590, embedding_dim=2048)

        self.word_vectors_matrix.weight.data = self.word_vectors
        self.word_vectors_linear = nn.Linear(2048, 768)
        self.juzi_word = nn.Linear(29184, 2048)
        self.linear_adapter1 = nn.Linear(768, 768)
    def kn_two_tiao(self, cruu_utt_id, text_utt_trans, kn_two_text, kn_similar_text):
        batch_size = len(text_utt_trans)  # 获取批次大小
        kn_two_array = np.empty((batch_size, 990))
        kn_similar_array = np.empty((batch_size, 12))
        cruu_utt_id = cruu_utt_id.tolist()
        # 遍历批处理中的每个句子特征向量
        for i, v1 in enumerate(cruu_utt_id):
            # 获取句子的索引，相似节点的索引，以及一跳节点的索引
            kn_similar_id = kn_similar_text[v1]
            if isinstance(kn_similar_id, str):
                if kn_similar_id.lower() == 'nan' or kn_similar_id == '':
                    kn_similar_id = []
                else:
                    kn_similar_id = kn_similar_id.split(', ')
            elif isinstance(kn_similar_id, float) and math.isnan(kn_similar_id):
                kn_similar_id = []
            else:
                kn_similar_id = []
            # kn_simi_list = kn_similar_id.split(', ')
            kn_similar_id = [int(x) for x in kn_similar_id]
            if len(kn_similar_id) < 24:
                # 指定目标长度
                target_length = 24
                # 使用列表推导式创建新列表，并补0
                simi_padded_list = [int(x) for x in kn_similar_id] + [0] * (target_length - len(kn_similar_id))
                # 转换为NumPy数组，并改变形状
                kn_simi_padded_array = np.array(simi_padded_list).reshape(1, -1)
                kn_similar_array[i] = kn_simi_padded_array
                small_value = -1e10
                kn_similar_array[i] = np.where(kn_similar_array[i] == 0, small_value, kn_similar_array[i])
            kn_two_id = kn_two_text[v1]
            if isinstance(kn_two_id, str):
                if kn_two_id.lower() == 'nan' or kn_two_id == '':
                    kn_two_list = []
                else:
                    kn_two_list = kn_two_id.split(', ')
            elif isinstance(kn_two_id, float) and math.isnan(kn_two_id):
                kn_two_list = []
            else:
                kn_two_list = []
            kn_twotiao_list = [int(x) for x in kn_two_list]
            if len(kn_twotiao_list) < 990:
                target_length = 990
                # 使用列表推导式创建新列表，并补0
                two_padded_list = [int(x) for x in kn_twotiao_list] + [0] * (target_length - len(kn_twotiao_list))
                # 转换为NumPy数组，并改变形状
                kn_two_padded_array = np.array(two_padded_list).reshape(1, -1)
                kn_two_array[i] = kn_two_padded_array
                small_value = -1e10
                kn_two_array[i] = np.where(kn_two_array[i] == 0, small_value, kn_two_array[i])
        return kn_similar_array, kn_two_array

    def kn_one_tiao(self, cruu_utt_id, text_utt_trans, kn_one_text, kn_similar_text):
        batch_size = len(text_utt_trans)  # 获取批次大小
        kn_one_array = np.empty((batch_size, 342))
        kn_similar_array = np.empty((batch_size, 24))
        cruu_utt_id = cruu_utt_id.tolist()
        # 遍历批处理中的每个句子特征向量
        for i, v1 in enumerate(cruu_utt_id):
            # 获取句子的索引，相似节点的索引，以及一跳节点的索引
            kn_similar_id = kn_similar_text[v1]
            if isinstance(kn_similar_id, str):
                if kn_similar_id.lower() == 'nan' or kn_similar_id == '':
                    kn_similar_id = []
                else:
                    kn_similar_id = kn_similar_id.split(', ')
            elif isinstance(kn_similar_id, float) and math.isnan(kn_similar_id):
                kn_similar_id = []
            else:
                kn_similar_id = []
            # kn_simi_list = kn_similar_id.split(', ')
            kn_similar_id = [int(x) for x in kn_similar_id]
            if len(kn_similar_id) < 24:
                # 指定目标长度
                target_length = 24
                # 使用列表推导式创建新列表，并补0
                simi_padded_list = [int(x) for x in kn_similar_id] + [0] * (target_length - len(kn_similar_id))
                # 转换为NumPy数组，并改变形状
                kn_simi_padded_array = np.array(simi_padded_list).reshape(1, -1)
                kn_similar_array[i] = kn_simi_padded_array
                small_value = -1e10
                kn_similar_array[i] = np.where(kn_similar_array[i] == 0, small_value, kn_similar_array[i])
            kn_one_id = kn_one_text[v1]
            if isinstance(kn_one_id, str):
                if kn_one_id.lower() == 'nan' or kn_one_id == '':
                    kn_one_list = []
                else:
                    kn_one_list = kn_one_id.split(', ')
            elif isinstance(kn_one_id, float) and math.isnan(kn_one_id):
                kn_one_list = []
            else:
                kn_one_list = []
            kn_onetiao_list = [int(x) for x in kn_one_list]
            if len(kn_onetiao_list) < 342:
                target_length = 342
                # 使用列表推导式创建新列表，并补0
                one_padded_list = [int(x) for x in kn_onetiao_list] + [0] * (target_length - len(kn_onetiao_list))
                # 转换为NumPy数组，并改变形状
                kn_one_padded_array = np.array(one_padded_list).reshape(1, -1)
                kn_one_array[i] = kn_one_padded_array
                small_value = -1e10
                kn_one_array[i] = np.where(kn_one_array[i] == 0, small_value, kn_one_array[i])
        return kn_similar_array, kn_one_array

    def two_tiao(self, text_utt_trans, word_one, one_100_text, cruu_utt_id):
        batch_size = len(text_utt_trans)  # 获取批次大小
        two_array = np.empty((batch_size, 200))
        # onee_array = np.empty((batch_size, 100))
        # similar_array = np.empty((batch_size, 5))
        cruu_utt_id = cruu_utt_id.tolist()
        # 遍历批处理中的每个句子特征向量
        for i, v1 in enumerate(text_utt_trans):
            # 假设sentence_features是一个二维数组，形状为（序列长度，维度）
            one_dim_features = v1.flatten()
            one_dim_features = self.juzi_word(one_dim_features)
            # 将句子的特征向量(1，2048)
            one_dim_features_juzi = one_dim_features.cpu().detach().numpy()
            # 获取句子的索引，相似节点的索引，以及一跳节点的索引
            # juzi_id = cruu_utt_id[i]
            one_text_list = one_100_text.tolist()
            one_list = one_text_list[i]
            one_id = [int(x) for x in one_list]
            twotiao_lists = set()  # 用集合来存储所有的一跳节点，保证去重
            for node_id in one_id:
                two_id = word_one[node_id]
                if isinstance(two_id, str):
                    if two_id.lower() == 'nan' or two_id == '':
                        twotiao_list = []
                    else:
                        twotiao_list = two_id.split(', ')
                elif isinstance(two_id, float) and math.isnan(two_id):
                    twotiao_list = []
                else:
                    twotiao_list = []
                twotiao_list = [int(x) for x in twotiao_list]
                # two_list = two_id.split(', ')
                # twotiao_lists = [int(x) for x in two_list]
                twotiao_lists.update(twotiao_list)  # 将当前节点的一跳节点添加到集合中
            # 将集合转换为列表，得到去重后的结果
            twotiao_list = list(twotiao_lists)
            if len(twotiao_list) > 200:
                word_vectors_matrix = self.word_vectors_matrix.weight.cpu().detach().numpy()
                input = torch.empty(8590, 2048)
                data = torch.zeros_like(input)
                two = data.cpu().detach().numpy()
                two[twotiao_list] = word_vectors_matrix[twotiao_list]
                sorted_indices = self.similarity_calculation(one_dim_features_juzi, two)
                two_hop_nodes = sorted_indices[:200]
                # one_hop_nodes=one_hop_nodes.tolist()
            else:
                two_hop_nodes = twotiao_list
            target_length = 200
            two_padded_list = [int(x) for x in two_hop_nodes] + [0] * (target_length - len(two_hop_nodes))
            two_padded_array = np.array(two_padded_list).reshape(1, -1)
            two_array[i] = two_padded_array
            # small_value = -1e10
            # two_array[i] = np.where(two_array[i] == 0, small_value, two_array[i])
        return two_array

    def simi_tiao(self, text_utt_trans, cruu_utt_id):
        batch_size = len(text_utt_trans)  # 获取批次大小
        similar_array = np.empty((batch_size, 5))  # 初始化相似度数组
        word_vectors_matrix_cpu = self.word_vectors_matrix.weight.cpu().detach().numpy()
        # 遍历批处理中的每个句子特征向量
        for i, v1 in enumerate(text_utt_trans):
            # 假设sentence_features是一个二维数组，形状为（序列长度，维度）
            one_dim_features = v1.flatten()
            one_dim_features = self.juzi_word(one_dim_features)
            # 将句子的特征向量(1，2048)
            one_dim_features_juzi = one_dim_features.cpu().detach().numpy()
            sorted_indices = self.similarity_calculation(one_dim_features_juzi, word_vectors_matrix_cpu)
            # 获取每个句子最相似的前五个词的索引
            top_five_indices = sorted_indices[:5]
            similar_array[i] = top_five_indices
        return similar_array

    '''def similarity_calculation(self, one_dim_features_juzi, one):
        num_cpu = np.dot(one_dim_features_juzi, one.T)
        denom = np.linalg.norm(one_dim_features_juzi) * np.linalg.norm(one, axis=1)  # 求模长的乘积
        res = num_cpu / denom
        res[np.isneginf(res)] = 0
        res = 0.5 + 0.5 * res
        sorted_indices = np.argsort(-res)
        return sorted_indices'''

    def similarity_calculation(self, one_dim_features_juzi, one):
        num_cpu = np.dot(one_dim_features_juzi, one.T)
        denom = np.linalg.norm(one_dim_features_juzi) * np.linalg.norm(one, axis=1)  # 求模长的乘积

        # 检查分母是否包含零或接近零的值，并进行处理
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.divide(num_cpu, denom, out=np.zeros_like(num_cpu), where=denom != 0)

        res[np.isneginf(res)] = 0
        res = 0.5 + 0.5 * res
        sorted_indices = np.argsort(-res)

        return sorted_indices

    # def one_tiao(self, text_utt_trans, one_text, similar_text, cruu_utt_id, one_100):
    def one_tiao(self, text_utt_trans, one_text, similar_text, cruu_utt_id):
        most_similar_indices = {}
        batch_size = len(text_utt_trans)  # 获取批次大小
        onee_array = np.empty((batch_size, 100))
        similar_array = np.empty((batch_size, 5))
        cruu_utt_id = cruu_utt_id.tolist()
        # 遍历批处理中的每个句子特征向量
        for i, v1 in enumerate(text_utt_trans):
            # 假设sentence_features是一个二维数组，形状为（序列长度，维度）
            one_dim_features = v1.flatten()
            one_dim_features = self.juzi_word(one_dim_features)
            # 将句子的特征向量(1，2048)
            one_dim_features_juzi = one_dim_features.cpu().detach().numpy()
            # 获取句子的索引，相似节点的索引，以及一跳节点的索引
            # juzi_id = cruu_utt_id[i]
            similar_text_list = similar_text.tolist()
            simi_list = similar_text_list[i]
            similar_id = [int(x) for x in simi_list]
            onetiao_lists = set()  # 用集合来存储所有的一跳节点，保证去重
            for node_id in similar_id:
                one_id = one_text[node_id]
                if isinstance(one_id, str):
                    if one_id.lower() == 'nan' or one_id == '':
                        onetiao_list = []
                    else:
                        onetiao_list = one_id.split(', ')
                elif isinstance(one_id, float) and math.isnan(one_id):
                    onetiao_list = []
                else:
                    onetiao_list = []
                onetiao_list = [int(x) for x in onetiao_list]
                # one_list = one_id.split(', ')
                # onetiao_lists = [int(x) for x in one_list]
                onetiao_lists.update(onetiao_list)  # 将当前节点的一跳节点添加到集合中
            # 将集合转换为列表，得到去重后的结果
            onetiao_list = list(onetiao_lists)
            if len(onetiao_list) > 100:
                word_vectors_matrix = self.word_vectors_matrix.weight.cpu().detach().numpy()
                input = torch.empty(8590, 2048)
                data = torch.zeros_like(input)
                one = data.cpu().detach().numpy()
                one[onetiao_list] = word_vectors_matrix[onetiao_list]
                sorted_indices = self.similarity_calculation(one_dim_features_juzi, one)
                one_hop_nodes = sorted_indices[:100]
                # one_hop_nodes=one_hop_nodes.tolist()
            else:
                one_hop_nodes = onetiao_list
            # one_100[juzi_id] = one_hop_nodes
            # 指定目标长度
            target_length = 100
            # 使用列表推导式创建新列表，并补0
            one_padded_list = [int(x) for x in one_hop_nodes] + [0] * (target_length - len(one_hop_nodes))
            # 转换为NumPy数组，并改变形状
            one_padded_array = np.array(one_padded_list).reshape(1, -1)
            # 将 padded_array 添加到结果数组中的第 i 行
            onee_array[i] = one_padded_array
            simi_padded_array = np.array(similar_id).reshape(1, -1)
            similar_array[i] = simi_padded_array
            # small_value = -1e10
            # onee_array[i] = np.where(onee_array[i] == 0, small_value, onee_array[i])
        return onee_array

    def jiedian_tezheng(self, most_similar_indices):
        concatenated_vectors_list = []
        zero_tensor = torch.zeros(2048).to('cuda')
        for node in most_similar_indices:
            vector_list = []
            for i in range(len(node)):
                if node[i] != 0:
                    index_tensor = torch.tensor([int(node[i])]).to('cuda')
                    vector_list.append(self.word_vectors_matrix(index_tensor).squeeze(0))
                else:
                    vector_list.append(zero_tensor)
            tensor_list_cuda = vector_list  # 所有的向量已经在 GPU 上，无需再转换
            tensor_list = torch.stack(tensor_list_cuda)
            concatenated_vectors_list.append(tensor_list)
        concatenated_vectors_cuda = [t.to('cuda') for t in concatenated_vectors_list]
        # 使用torch.stack()将列表中的张量连接起来
        concatenated_vectors = torch.stack(concatenated_vectors_cuda, dim=0)

        return concatenated_vectors

    def kn_jiedian_tezheng(self, kn_similar_kn_indices):
        concatenated_vectors_list = []
        zero_tensor = torch.full((2048,), -1e-10).to('cuda')
        for node in kn_similar_kn_indices:
            vector_list = []
            for i, value in enumerate(node):
                if math.isnan(value):
                    vector_list.append(zero_tensor)
                elif value != -10000000000.0:
                    n = int(value)
                    # 调用 Embedding 对象获取索引对应的词向量
                    n_tensor = self.kn_word_vectors_tensor(torch.tensor([n]).to('cuda'))
                    n_tensor = n_tensor.squeeze()
                    vector_list.append(n_tensor)
                else:
                    vector_list.append(zero_tensor)

            tensor_list_cuda = [t.to('cuda') for t in vector_list]

            tensor_list = torch.stack(tensor_list_cuda)

            # 将当前的 tensor_list 添加到列表中
            concatenated_vectors_list.append(tensor_list)
        concatenated_vectors_cuda = [t.to('cuda') for t in concatenated_vectors_list]
        # 使用torch.stack()将列表中的张量连接起来
        concatenated_vectors = torch.stack(concatenated_vectors_cuda, dim=0)
        return concatenated_vectors


    '''def forward(self, batch_text_input_ids=None, batch_text_input_mask=None, batch_text_sep_mask=None, 
                        audio_inputs=None, audio_mask=None, vision_inputs=None, new_vision_mask=None, 
                        batchUtt_in_dia_idx=None):'''

    def forward(self, batch_text_input_ids=None, batch_text_input_mask=None, batch_text_sep_mask=None,
            audio_inputs=None, audio_mask=None, vision_inputs=None, vision_mask=None, batch_vis_emo=None, batchUtt_in_dia_idx=None, split=None):

        if self.text_pretrained_model == 'roberta':
            #<s>utt_1</s></s>utt_2</s></s>utt_3</s>...
            outputs = self.roberta(batch_text_input_ids, batch_text_input_mask)
        else:
            #[CLS]utt_1[SEP]utt_2[SEP]utt_3[SEP]...
            outputs = self.bert(batch_text_input_ids, batch_text_input_mask)

        text_pretrained_model_out = outputs[0]  # (num_dia, max_sequence_len, 1024)
        text_utt_linear = self.text_linear(text_pretrained_model_out) #(num_dia, max_sequence_len, hidden_size)
        
        '''
        Extract word-level textual representations for each utterance.
        '''
        utt_batch_size = vision_inputs.shape[0]

        batch_text_feat_update = torch.zeros((utt_batch_size, self.get_text_utt_max_lens, text_utt_linear.shape[-1])).cuda()  #batch_size, max_utt_len, hidden_size
        batch_text_sep_mask_update = torch.zeros((utt_batch_size, self.get_text_utt_max_lens)).cuda() #batch_size, max_utt_len

        for i in range(utt_batch_size):
            curr_utt_in_dia_idx = batchUtt_in_dia_idx[i] #the position of the target utterance in the current dialogue.
            curr_dia_mask = batch_text_sep_mask[i]
            each_utt_index = []

            for index, value in enumerate(list(curr_dia_mask)): 
                if value == 1:  
                    each_utt_index.append(index)  #record the index position of the first </s> or [SEP] token for each utterance.
                    if curr_utt_in_dia_idx  == 0:  
                        '''the current utterance is at the 0th position in the dialogue.'''
                        curr_utt_len = index-1   #remove the starting <s> and ending </s> tokens, or remove the starting [CLS] and ending [SEP] tokens.
                        if curr_utt_len > self.get_text_utt_max_lens:
                            curr_utt_len = self.get_text_utt_max_lens
                        batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][1:curr_utt_len+1] #从<s>或者[CLS]之后开始
                        batch_text_sep_mask_update[i][:curr_utt_len] = 1
                        break
                    elif curr_utt_in_dia_idx >0 and curr_utt_in_dia_idx + 1 == len(each_utt_index): 
                        curr_ut_id = len(each_utt_index) -1 #1 
                        pre_ut_id = len(each_utt_index) - 2 #0
                        curr_ut = each_utt_index[curr_ut_id]
                        pre_ut = each_utt_index[pre_ut_id]
                        if self.text_pretrained_model == 'roberta':
                            curr_utt_len = curr_ut - pre_ut - 2  #remove </s> and <s>
                        else:
                            curr_utt_len = curr_ut - pre_ut - 1 #remove [SEP]

                        if curr_utt_len > self.get_text_utt_max_lens:
                            curr_utt_len = self.get_text_utt_max_lens
                        if self.text_pretrained_model == 'roberta':
                            batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][pre_ut+2:pre_ut+2+curr_utt_len]  #从</s><s>之后开始
                        else:
                            batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][pre_ut+1:pre_ut+1+curr_utt_len]  #从[SEP]之后开始
                        batch_text_sep_mask_update[i][:curr_utt_len] = 1
                        break
        #for memory 
        del text_utt_linear, batch_text_sep_mask
        text_utt_trans = batch_text_feat_update
        # 所有结点的一跳节点
        word = "/home/stu4/project/COSIKE/graph/meld_oneid.csv"
        #word = "/home/stu4/project/COSIKE/graph/quanju/meld_quanju_oneid.csv"
        #word = "/home/stu4/project/COSIKE/graph/jubu/meld_jubu_oneid.csv"
        df_word=pd.read_csv(word, encoding='gbk')
        wordid=df_word['word_id']
        word_one_id=df_word['word_one_id']
        word_one = dict(zip(wordid, word_one_id))

        # 常识节点，匹配的节点，一跳节点，二跳节点
        kn_one_path = "/home/stu4/project/COSIKE/graph/knowledge/" + split + "_kn_text.csv"
        df_kn = pd.read_csv(kn_one_path, encoding='gbk')
        sentid = df_kn['sent_id']
        kn_one_word_column = df_kn['one_hop']
        #kn_two_word_column = df_kn['二跳节点']
        kn_similar_word_column = df_kn['id']
        # 将列转换为字典
        kn_one_text = dict(zip(sentid, kn_one_word_column))
        #kn_two_text = dict(zip(sentid, kn_two_word_column))
        kn_similar_text = dict(zip(sentid, kn_similar_word_column))

        cruu_utt_id = batchUtt_in_dia_idx
        #输入常识图的节点以及拿到特征
        kn_similar, kn_one = self.kn_one_tiao(cruu_utt_id, text_utt_trans, kn_one_text, kn_similar_text)
        #kn_similar, kn_two = self.kn_two_tiao(cruu_utt_id, text_utt_trans, kn_two_text, kn_similar_text)
        kn_simi_tezheng=self.kn_jiedian_tezheng(kn_similar)  #常识的相似节点的特征（max:6, min:0）
        kn_one_tezheng = self.kn_jiedian_tezheng(kn_one)  # 常识的一跳节点的特征 (max:292 )
        #kn_two_tezheng=self.kn_jiedian_tezheng(kn_two)  #常识的二跳节点的特征(max:691)
        kn_simi_tezheng = self.word_vectors_linear(kn_simi_tezheng)
        kn_one_tezheng = self.word_vectors_linear(kn_one_tezheng)


        cruu_utt_id = batchUtt_in_dia_idx
        #输入全局图和局部图的一跳节点
        #most_one_indices,similar_array=self.one_tiao(text_utt_trans, one_text, similar_text, cruu_utt_id)
        similar_array = self.simi_tiao(text_utt_trans, cruu_utt_id)
        most_one_indices = self.one_tiao(text_utt_trans, word_one, similar_array, cruu_utt_id)
        #most_two_indices=self.two_tiao(text_utt_trans, word_one, most_one_indices, cruu_utt_id)
        #根据最相似的节点得到100个最相似的1跳节点的特征
        one_tezheng=self.jiedian_tezheng(most_one_indices)
        one_tezheng_768=self.word_vectors_linear(one_tezheng)
        #最相似的5个节点的特征
        simi_tezheng = self.jiedian_tezheng(similar_array)
        simi_tezheng_768 = self.word_vectors_linear(simi_tezheng)
        #最相似的200个二跳节点的特征
        #two_tezheng = self.jiedian_tezheng(most_two_indices)
        #two_tezheng_768=self.word_vectors_linear(two_tezheng)
        # text_q_word_one=torch.tanh(self.linear_adapter1(torch.tanh(self.linear_adapter1(text_q_one)) + text_utt_trans))
        #text_q_word_two = torch.tanh(self.linear_adapter1(torch.tanh(self.linear_adapter1(text_q_two)) + text_utt_trans))

        text_q_knsimi, _ = self.multi_head_attention_text_gra(text_utt_trans, kn_simi_tezheng, kn_simi_tezheng)
        text_q_kn1, _ = self.multi_head_attention_text_gra(text_q_knsimi, kn_one_tezheng, kn_one_tezheng)
        #text_q_kn2, _ = self.multi_head_attention_text_gra(text_q_kn1, kn_two_tezheng, kn_two_tezheng)
        text_kn_qj = torch.tanh(self.linear_adapter1(torch.tanh(self.linear_adapter1(text_q_kn1)) + text_utt_trans))

        #text_q_simi, _ = self.multi_head_attention_text_gra(text_utt_trans, simi_tezheng_768, simi_tezheng_768)   #句子与最相似的5个节点融合
        text_q_simi, _ = self.multi_head_attention_text_gra(text_kn_qj, simi_tezheng_768, simi_tezheng_768)  # 句子与最相似的5个节点融合
        text_q_one, _ = self.multi_head_attention_text_gra(text_q_simi, one_tezheng_768, one_tezheng_768)        #句子与最相似的5个节点融合后，与100个一跳再融合
        #text_q_two,_ = self.multi_head_attention_text_gra(text_q_one, two_tezheng_768, two_tezheng_768)
        #句子与5个节点融合，和与100个融合，再与200个融合
        
        text_kn_qj = torch.tanh(self.linear_adapter1(torch.tanh(self.linear_adapter1(text_q_one)) + text_utt_trans))
        #text_kn_qj = torch.tanh(self.linear_adapter1(torch.tanh(self.linear_adapter1(text_q_two)) + text_utt_trans))

        batch_text_feat_update = text_kn_qj


        '''audio modality'''
        #Input dim: (batch_size, Max_utt_len, pretrained_wav2vec_dim)
        audio_extended_utt_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        audio_extended_utt_mask = (1.0 - audio_extended_utt_mask) * -10000.0
        audio_emb_linear = self.audio_linear(audio_inputs) 
        audio_utt_trans = self.audio_utt_transformer(audio_emb_linear, audio_extended_utt_mask)    #(batch_size, utt_max_lens, self.hidden_size)
        
        '''visual modality'''
        #Input dim: (batch_size, Max_utt_len, pretrained_IncepRes_dim + 7)
        '''vision_extended_utt_mask = new_vision_mask.unsqueeze(1).unsqueeze(2)
        vision_extended_utt_mask = (1.0 - vision_extended_utt_mask) * -10000.0  
        vision_emb_linear = self.vision_linear(vision_inputs) 
        vision_utt_trans = self.vision_utt_transformer(vision_emb_linear, vision_extended_utt_mask) #(batch_size, utt_max_lens, self.hidden_size)'''


        '''visual modality'''
        vision_extended_utt_mask = vision_mask.unsqueeze(1).unsqueeze(2)
        vision_extended_utt_mask = (1.0 - vision_extended_utt_mask) * -10000.0
        #vision_emb_linear = self.vision_linear(vision_inputs)
        #vision_utt_trans = self.vision_utt_transformer(vision_emb_linear, vision_extended_utt_mask)  # (batch_size, utt_max_lens, self.hidden_size)
        # Multiply emotion probabilities with the weight matrix
        emo_features = torch.matmul(batch_vis_emo, self.W)
        # Concatenate emotion features with vision features
        combined_vision_features = torch.cat((vision_inputs, emo_features), dim=-1)
        vision_emb_linear = self.vision_linear(combined_vision_features)
        vision_utt_trans = self.vision_utt_transformer(vision_emb_linear, vision_extended_utt_mask)


        # cross-modality
        batch_text_feat_update = batch_text_feat_update.transpose(0,1)
        audio_utt_trans = audio_utt_trans.transpose(0,1)
        text_crossAudio_att = self.CrossModalTrans_TA(batch_text_feat_update, audio_utt_trans, audio_utt_trans) #(max_textUtt_len,  batch_size, self.hidden_size)
        audio_crossText_att = self.CrossModalTrans_TA(audio_utt_trans, batch_text_feat_update, batch_text_feat_update) #(max_audioUtt_len, batch_size, self.hidden_size)
        textaudio_cross_feat = torch.concat((text_crossAudio_att, audio_crossText_att),dim=0) #(max_textUtt_len + max_audioUtt_len, batch_size, self.hidden_size)

        vision_utt_trans = vision_utt_trans.transpose(0,1)
        vision_crossTeAud_att = self.CrossModalTrans_TA_V(vision_utt_trans, textaudio_cross_feat, textaudio_cross_feat) #(max_visionUtt_len, batch_size, self.hidden_size)
        texaudi_crossVis_att = self.CrossModalTrans_TA_V(textaudio_cross_feat, vision_utt_trans, vision_utt_trans) #(max_textUtt_len + max_audioUtt_len, batch_size, self.hidden_size)
        final_utt_feat = torch.concat((texaudi_crossVis_att, vision_crossTeAud_att),dim=0) #(max_textUtt_len + max_audioUtt_len + max_visionUtt_len, batch_size, self.hidden_size)
        final_utt_feat = final_utt_feat.transpose(0,1) 
        T_A_utt_mask = torch.concat((batch_text_sep_mask_update, audio_mask),dim=1)
        #final_utt_mask = torch.concat((T_A_utt_mask, new_vision_mask),dim=1)
        final_utt_mask = torch.concat((T_A_utt_mask, vision_mask), dim=1)

        multimodal_out, _ = self.attention(final_utt_feat, final_utt_mask) #（batch_size, self.hidden_size）

        # classify
        multimodal_output = self.dropout(multimodal_out)  
        logits = self.classifier(multimodal_output)
        return logits


'''vision model'''
class meld_utt_transformer(nn.Module):
    def __init__(self, args):
        super(meld_utt_transformer, self).__init__()

        num_labels = args.num_labels
        self.modality_origin_emb = args.vision_featExtr_dim
        self.modality_utt_Transformernum = args.vision_utt_Transformernum
        self.get_utt_max_lens = args.get_vision_utt_max_lens   

        self.hidden_size = args.hidden_size
        self.hidden_dropout_prob = args.hidden_dropout_prob   
        self.modality_linear = nn.Linear(self.modality_origin_emb, self.hidden_size)
        self.utt_transformer = MELDTransEncoder(args, self.modality_utt_Transformernum, self.get_utt_max_lens, self.hidden_size) 
        self.attention = AdditiveAttention(self.hidden_size, self.hidden_size)  
        self.mm_dropout = nn.Dropout(self.hidden_dropout_prob)  
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, inputs=None, utt_mask=None):
        '''
        input_shape: (batch_size, utt_max_lens, featureExtractor_dim)  
        utt_mask_shape: (batch_size, utt_max_lens)
        '''
        extended_utt_mask = utt_mask.unsqueeze(1).unsqueeze(2)
        extended_utt_mask = extended_utt_mask.to(dtype=next(self.parameters()).dtype)
        extended_utt_mask = (1.0 - extended_utt_mask) * -10000.0  
        modality_emb_linear = self.modality_linear(inputs)
        modality_utt_trans = self.utt_transformer(modality_emb_linear, extended_utt_mask)  #(batch_size, utt_max_lens, self.hidden_size)
        utt_trans_attention, _ = self.attention(modality_utt_trans, utt_mask) #(batch_size, self.hidden_size)
        outputs = self.mm_dropout(utt_trans_attention)  
        logits = self.classifier(outputs)  #(batch_size, 7)

        return logits
