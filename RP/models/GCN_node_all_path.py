import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
# import torch_geometric as tg
from pytorch_pretrained_bert import BertModel
# from models.transformer.Models import Encoder
from torch.nn.utils.rnn import pad_sequence


# naive graph convolutional networks
class GraphConv(nn.Module):
	def __init__(self, input_dim,edge_dim, output_dim, bias=False):
		super(GraphConv, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.act_func = F.relu
		self.weights_edge = nn.Parameter(torch.FloatTensor(edge_dim, output_dim))
		self.weights_node = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(output_dim))
		else:
			self.register_parameter('bias', None)
		self.init()
	
	def init(self):
		nn.init.xavier_uniform_(self.weights_edge.data)
		nn.init.xavier_uniform_(self.weights_node.data)
	
	def forward(self, inputs,edge_inputs, adjacency_matrix):
		'''
		inputs: shape = [num_entity, embedding_dim]
		'''
		outputs_edge = torch.einsum('ijk,kp->ijp',edge_inputs,self.weights_edge)
		#outputs_edge = self.linear_edge(edge_inputs)
		#print ('weight_edge',self.weights_edge)
		#outputs_edge = torch.bmm(edge_inputs,self.weights_edge.unsqueeze(0).expand(self.input_dim))
		outputs_edge = torch.mean(outputs_edge,dim = 1)
		outputs_node = torch.chain_matmul(adjacency_matrix, inputs, self.weights_node)
		outputs = outputs_edge + outputs_node
		# support = torch.mm(inputs, self.weights)
		# outputs = torch.spmm(inputs, support)
		if self.bias is not None:
			outputs += self.bias
		node_weight = torch.sum(adjacency_matrix,1)
		node_weight_zero = torch.eq(node_weight,0).float()
		node_weight = (node_weight + node_weight_zero).unsqueeze(1).expand_as(outputs)
		return outputs / node_weight

class GraphConvolution(nn.Module):
	def __init__(self, layer_num, input_dim, output_dim, bias=False):
		super(GraphConvolution, self).__init__()
		self.input_dim = input_dim
		self.layer_num = layer_num
		hidden_dim = output_dim
		graph_hidden_dim = int(hidden_dim/layer_num)
		self.gcn_dropout = nn.Dropout(0.2)
		self.graphconv = nn.ModuleList([GraphConv(input_dim + graph_hidden_dim * i, input_dim , graph_hidden_dim) for i in range (layer_num)])
		self.linear_layer = nn.Linear(hidden_dim,output_dim)
	
	def forward(self, node_feat,edge_feat,adj_matrix):
		'''
		inputs: shape = [num_entity, embedding_dim]
		'''
		outputs = node_feat
		output_list = []
		cache_list = [outputs]
		for l in range (self.layer_num):
			graph_out = torch.relu(self.graphconv[l](outputs,edge_feat,adj_matrix))
			cache_list.append(graph_out)
			outputs = torch.cat(cache_list, dim=-1)
			output_list.append(self.gcn_dropout(graph_out))
		node_feat_output = torch.cat(output_list,dim = -1)
		node_feat_output = node_feat_output + node_feat

		final_output = self.linear_layer(node_feat_output)

		return final_output

class MultiGraphConvolution(nn.Module):
	def __init__(self, layer_num, head_num, input_dim, output_dim, bias=False):
		super(MultiGraphConvolution, self).__init__()
		self.input_dim = input_dim
		self.layer_num = layer_num
		self.head_num = head_num
		hidden_dim = output_dim
		graph_hidden_dim = int(hidden_dim/layer_num)
		self.gcn_dropout = nn.Dropout(0.2)
		self.graphconv = nn.ModuleList()
		for i in range (head_num):
			for j in range (layer_num):
				self.graphconv.append(GraphConv(input_dim + graph_hidden_dim * j ,input_dim, graph_hidden_dim))
		self.linear_layer = nn.Linear(hidden_dim * head_num,output_dim)
	
	def forward(self, node_feat,edge_feat,adj_matrix_list):
		'''
		inputs: shape = [num_entity, embedding_dim]
		'''
		feat_head_list = []
		for h in range (self.head_num):
			outputs = node_feat
			output_list = []
			cache_list = [outputs]
			for l in range (self.layer_num):
				index = h*self.layer_num + l
				graph_out = torch.relu(self.graphconv[index](outputs,edge_feat,adj_matrix_list[h]))
				cache_list.append(graph_out)
				outputs = torch.cat(cache_list, dim=-1)
				output_list.append(self.gcn_dropout(graph_out))
			node_feat_output = torch.cat(output_list,dim = -1)
			node_feat_output = node_feat_output + node_feat

			feat_head_list.append(node_feat_output)
		
		feat_head_list = torch.cat(feat_head_list,-1)
		final_output = self.linear_layer(feat_head_list)

		return final_output

class MultiHeadAttention(nn.Module):
	def __init__(self, head_num, att_size, dropout=0.1):
		super(MultiHeadAttention, self).__init__()
		assert att_size % head_num == 0

		self.hidden_size = att_size // head_num
		self.head_num = head_num
		self.linears_q = nn.ModuleList([nn.Linear(att_size, self.hidden_size) for _ in range (head_num)])
		self.linears_k = nn.ModuleList([nn.Linear(att_size, self.hidden_size) for _ in range (head_num)])
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, node_feat, mask = None):
		att_list = []
		for h in range(self.head_num):
			query = self.linears_q[h](node_feat)
			key = self.linears_q[h](node_feat).transpose(0,1)
			att = torch.softmax(torch.mm(query,key)/math.sqrt(self.hidden_size),dim = -1)
			if self.dropout is not None:
				att = self.dropout(att)
			att_list.append(att)
		return att_list

class GATAttention(nn.Module):
	def __init__(self, att_input_dim,hidden_dim, dropout=0.1):
		super(GATAttention, self).__init__()
		
		self.linear_node_h = nn.Linear (att_input_dim,hidden_dim)
		self.linear_node_t = nn.Linear (att_input_dim,hidden_dim)
		self.linear_edge_r = nn.Linear (att_input_dim,hidden_dim)
		self.wt = nn.Linear (hidden_dim * 3,1)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, node_feat, edge_feat, mask = None):
		node_num = node_feat.size(0)
		node_feat_h = node_feat.unsqueeze(0).expand(node_num,node_num,-1)
		node_feat_t = node_feat.unsqueeze(0).expand(node_num,node_num,-1)

		node_att_h = self.linear_node_h(node_feat_h) 
		node_att_t = self.linear_node_t(node_feat_t)
		edge_att_r = self.linear_edge_r(edge_feat)
		energy_att = self.wt(torch.cat([node_att_h,node_att_t,edge_att_r],-1)).squeeze(-1)
		if mask is not None:
			energy_att.masked_fill(mask.bool(),-100000.0)
		att = torch.softmax(energy_att,dim = -1)
		if self.dropout is not None:
			att = self.dropout(att)
		return att


# class GATAttention(nn.Module):
# 	def __init__(self, head_num, att_input_size, dropout=0.1):
# 		super(GATAttention, self).__init__()
# 		assert att_input_size % head_num == 0
# 		self.head_num = head_num
# 		self.hidden_size = att_input_size // head_num		
# 		self.linear_node_h = nn.ModuleList([nn.Linear (att_input_size,self.hidden_size) for _ in range (head_num)])
# 		self.linear_node_t = nn.ModuleList([nn.Linear (att_input_size,self.hidden_size) for _ in range (head_num)])
# 		self.linear_edge_r = nn.ModuleList([nn.Linear (att_input_size,self.hidden_size) for _ in range (head_num)])
# 		self.wt = nn.ModuleList([nn.Linear (self.hidden_size * 3,1) for _ in range (head_num)])
# 		self.dropout = nn.Dropout(p=dropout)

# 	def forward(self, node_feat, edge_feat, mask = None):
# 		node_num = node_feat.size(0)
# 		node_feat_h = node_feat.unsqueeze(0).expand(node_num,node_num,-1)
# 		node_feat_t = node_feat.unsqueeze(0).expand(node_num,node_num,-1)
# 		att_list = []
# 		for h in range(self.head_num):
# 			node_att_h = self.linear_node_h[h](node_feat_h) 
# 			node_att_t = self.linear_node_t[h](node_feat_t)
# 			edge_att_r = self.linear_edge_r[h](edge_feat)
# 			energy_att = self.wt[h](torch.cat([node_att_h,node_att_t,edge_att_r],-1)).squeeze(-1)
# 			if mask is not None:
# 				energy_att.masked_fill(mask,-100000.0)
# 			att = torch.softmax(energy_att,dim = -1)
# 			if self.dropout is not None:
# 				att = self.dropout(att)
# 			att_list.append (att)
# 		if len(att_list) == 1:
# 			return att_list[0]
# 		return att_list


class WordAttention(nn.Module):
	def __init__(self,input_dim,hidden_dim,position_dim):
		super(WordAttention, self).__init__()
		self.attention_sent = nn.Linear(input_dim,hidden_dim)
		self.attention_pos = nn.Linear(position_dim,hidden_dim)
		self.attention_all = nn.Linear(hidden_dim,1)
	
	def forward (self,att_padding_matrix,context_before_att,dis_embedding):
		sent_feat = self.attention_sent(context_before_att)   #n * n * sent_num * sent_len * hidden_dim
		dis_feat = self.attention_pos(dis_embedding)
		# sent_feat = sent_feat + dis_feat

		all_feat = self.attention_all(torch.tanh(sent_feat + dis_feat)) #n * n * sent_num * sent_len * 1
		# all_feat = self.attention_all(nn.functional.leaky_relu(sent_feat + dis_feat, negative_slope=0.2)) #n * n * sent_num * sent_len * 1
		att_padding_matrix = att_padding_matrix.expand_as(all_feat)
		all_feat = all_feat.masked_fill(att_padding_matrix.bool(), -100000.0)	#将att_padding_matrix 为1位置对应的all——feat 元素改为-100000.0
		att_matrix = nn.functional.softmax(all_feat,dim = 3).expand_as(context_before_att)
		context_after_att = torch.sum(att_matrix * context_before_att,dim =3)
		
		return context_after_att


class SentenceAttention(nn.Module):
	def __init__(self,input_dim,hidden_dim):
		super(SentenceAttention, self).__init__()
		self.attention_sent = nn.Linear(input_dim,hidden_dim)
		self.attention_pos = nn.Linear(input_dim,hidden_dim)
		self.attention_all = nn.Linear(hidden_dim,1)
	
	def forward (self,att_padding_matrix,context_word_att,node_embedding):
		sent_feat = self.attention_sent(context_word_att)   #n * n * sent_num * hidden_dim
		dis_feat = self.attention_pos(node_embedding)
		# sent_feat = sent_feat + dis_feat
		all_feat = self.attention_all(torch.tanh(sent_feat + dis_feat)) #n * n * sent_num * 1
		# all_feat = self.attention_all(nn.functional.leaky_relu(sent_feat + dis_feat, negative_slope=0.2)) #n * n * sent_num * 1

		sent_num = att_padding_matrix.sum(dim = 2)
		
		#att_padding_matrix = att_padding_matrix.expand_as(all_feat)
		all_feat = all_feat.masked_fill(att_padding_matrix.bool(), -100000.0)   # -np.inf

		# att_matrix = torch.sigmoid(all_feat).expand_as(context_word_att)
		att_matrix = torch.relu(all_feat).expand_as(context_word_att)
		context_after_att = torch.sum(att_matrix * context_word_att,dim = 2)/ (sent_num + 1e-10)
		
		return context_after_att



class GCN_node_all_path(nn.Module):
	def __init__(self, config):
		super(GCN_node_all_path, self).__init__()
		self.config = config
		print('model: GCGCN_node_all_path')
		print('原本的GCGCN模型，抽取所有简单路径，拼接路径上的结点表示进行预测,average pool')
		# print('在计算两个层次的attention时，用relu，改了两处')

		#word_vec_size = config.data_word_vec.shape[0]
		#self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])        #词向量矩阵
		#self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
		word_vec_size = 768
		self.bert = BertModel.from_pretrained('./bert/bert-base-uncased')

		#self.word_emb.weight.requires_grad = False                  #词向量不更新？？？？
		self.use_entity_type = True
		self.use_coreference = True
		self.use_type_feat = True
		self.use_distance = True

		self.dropout = nn.Dropout(0.2)

		# performance is similar with char_embed
		# self.char_emb = nn.Embedding(config.data_char_vec.shape[0], config.data_char_vec.shape[1])
		# self.char_emb.weight.data.copy_(torch.from_numpy(config.data_char_vec))

		# char_dim = config.data_char_vec.shape[1]
		# char_hidden = 100
		# self.char_cnn = nn.Conv1d(char_dim,  char_hidden, 5)

		hidden_size = 128
		input_size = word_vec_size
		if self.use_entity_type:                                                        #类型特征向量矩阵
			input_size += config.entity_type_size
			self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

		if self.use_coreference:                                                         #实体特征
			input_size += config.coref_size
			# self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
			self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

		# input_size += char_hidden

		#self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)   #包含一个rnn编码层
		self.layerNum = 4
		self.headNum = 4
		# self.transformer = Encoder(self.layerNum,self.headNum,input_size, input_size, input_size, useAtt = False,use_cuda = True)
		self.get_weighted_adj_matrix = GATAttention (hidden_size,hidden_size)
		self.get_adj_matrix =nn.ModuleList([MultiHeadAttention(self.headNum,hidden_size) for _ in range (self.config.graph_hop-1)])
		#self.get_weighted_adj_matrix =nn.ModuleList([GATAttention (self.headNum,hidden_size) for _ in range (self.config.graph_hop)])
		self.graphcnn = nn.ModuleList()  
		for layer in range (self.config.graph_hop):
			if layer == 0:
				self.graphcnn.append(GraphConvolution(self.layerNum,hidden_size,hidden_size))                             #包含一个图卷积
			else:
				self.graphcnn.append(MultiGraphConvolution(self.layerNum,self.headNum,hidden_size,hidden_size))
		
		#self.linear_re = nn.Linear(hidden_size*2, hidden_size)                                        #包含一个线性层
		self.linear_re = nn.Linear(input_size,hidden_size)
		self.word_attention = nn.ModuleList([WordAttention (hidden_size,hidden_size,position_dim = config.dis_size) for _ in range(self.config.graph_hop)])
		self.sentence_attention = nn.ModuleList([SentenceAttention (hidden_size,hidden_size) for _ in range(self.config.graph_hop)])


		self.linear_word_att =  nn.ModuleList([nn.Linear(hidden_size*2, hidden_size) for _ in range(self.config.graph_hop)])                                        #word_attention后的线性层
		self.linear_sentence_att =  nn.ModuleList([nn.Linear(hidden_size*2, hidden_size) for _ in range(self.config.graph_hop)])                                        #sentence_attention后的线性层

		# if self.use_type_feat:
		# 	self.dense_layer = nn.Linear(hidden_size*(config.graph_hop+1)+ config.dis_size+config.entity_type_size ,hidden_size)   #config.graph_hop+1
		# else:
		# 	self.dense_layer = nn.Linear(hidden_size*(config.graph_hop+1)+config.dis_size ,hidden_size)


		if self.use_type_feat:
			self.dense_layer = nn.Linear(hidden_size*(config.graph_hop+1)+config.entity_type_size ,hidden_size)   #config.graph_hop+1
		else:
			self.dense_layer = nn.Linear(hidden_size*(config.graph_hop+1) ,hidden_size)


		# self.bili_layer_01 = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)
		# self.classification_layer_01 = nn.Linear(hidden_size*2,config.relation_num)
		self.linear_cls = nn.Linear(word_vec_size,hidden_size)
		#self.bili_layer_multi = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num-1)
		#self.classification_layer_multi = nn.Linear(hidden_size*2,config.relation_num-1)

		#self.dense_layer_multi = nn.Linear(hidden_size,hidden_size)

		if self.use_distance:
			self.dis_embed = nn.Embedding(config.dis_num, config.dis_size)#, padding_idx=10)

		hidden_size_2 = hidden_size*(config.graph_hop+1)+config.entity_type_size

		# self.rnn_path = EncoderLSTM(hidden_size_2, hidden_size_2, 1, True, True, 1 - config.keep_prob, False)
		self.bilstm_path = nn.LSTM(hidden_size_2,hidden_size_2,num_layers =2,bidirectional =True)
		self.linear_path = nn.Linear(hidden_size_2 * 2,hidden_size_2)
		# self.linear_path_class = nn.Linear(hidden_size_2,config.relation_num)
		self.linear_path_output = nn.Linear(hidden_size_2,hidden_size)
		self.linear_context = nn.Linear(hidden_size,hidden_size_2)
		self.calss_linear = nn.Linear(hidden_size*4,hidden_size)
		self.linaer_out = nn.Linear(hidden_size,config.relation_num)

	def forward(self, document,document_ner,document_pos,adj_matrix,sen_matrix,pos_matrix_h,pos_matrix_t,node_pos,node_type,node_relative_pos,is_train,node_pairs_path_dict,all_paths_list):
		# para_size, char_size, bsz = context_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)
		# context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
		# context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)

		#doc = self.word_emb(document)
		#print (document)
		doc,_ = self.bert(document.unsqueeze(0),output_all_encoded_layers = False)
		doc = doc.squeeze(0)
		cls_feat = doc[0,:]
		
		if self.use_coreference:
			doc = torch.cat([doc, self.entity_embed(document_pos)], dim=-1)

		if self.use_entity_type:
			doc = torch.cat([doc, self.ner_emb(document_ner)], dim=-1)

		# sent = torch.cat([sent, context_ch], dim=-1)
		# print (doc.size())                      [197,140]
		#context_output = self.rnn(doc.unsqueeze(0), doc.size(0))
		context_output = torch.tanh(self.linear_re(doc.unsqueeze(0)))

		# print (context_output.size())               #[1, context_len, 128]
		# print (pos_matrix_h.size())                 [17, 17, 3, 197]
		# print (node_pos.size())                     #[node_num, context_len]
		# print (sen_matrix.size())                   [17, 17, 3, 197]
		# initialize the node features
		node_num, max_sen_num, document_length = sen_matrix.size(0), sen_matrix.size(2), sen_matrix.size(3)
		hidden_dim = context_output.size(-1)
		
		#context_output = torch.relu(self.linear_rnn(doc.unsqueeze(0)))
		# print(node_pos.unsqueeze(2).expand(-1,-1,hidden_dim).shape)	#[node_num,context_len,128]
		# print(context_output.expand(node_num,-1,-1).shape)		#[node_num,context_len,128]
		node_feat = node_pos.unsqueeze(2).expand(-1,-1,hidden_dim) * context_output.expand(node_num,-1,-1)	#[node_num,context_len,128]
		node_feat = node_feat.sum(dim = 1)     #此处得到特征， 维度 n*hidden_dim 作为邻接矩阵的初始
		# print('node_feat',node_feat.shape)		#[node_num * 128]
		#print (sen_matrix.size())                    [17, 17, 3, 197]
		word_att_padding_matrix = ~ sen_matrix.unsqueeze(4)
		context_before_att = context_output.unsqueeze(0).unsqueeze(0).expand (node_num,node_num,max_sen_num,-1,-1)
		#sentence_level_attention
		sent_att_padding_matrix = ~ sen_matrix[:,:,:,0:1]		#:表示那个维度切取完整对象
		#distance embedding
		dis_embedding_h = self.dis_embed(pos_matrix_h)
		dis_embedding_t = self.dis_embed(pos_matrix_t)
		node_relative_pos_h = self.dis_embed(self.config.dis_plus + node_relative_pos)   #[17,17,128]
		node_relative_pos_t = self.dis_embed(self.config.dis_plus - node_relative_pos)   #[17,17,128]
		#print (context_before_att.size())            [17, 17, 3, 197, 128]
		
		# rel_layer_feat = [node_feat.unsqueeze(0).expand(node_num,-1,-1) * node_feat.unsqueeze(1).expand(-1,node_num,-1)]
		node_feats = [node_feat]
		for i in range(self.config.graph_hop):
			# two attentions
			
			#word_level_attention
			#print (word_att_padding_matrix.size(),context_before_att.size(),dis_embedding_h.size())
			context_word_att_h = self.word_attention[i] (word_att_padding_matrix,context_before_att,dis_embedding_h)
			context_word_att_t = self.word_attention[i] (word_att_padding_matrix,context_before_att,dis_embedding_t)

			context_word_att = torch.cat([context_word_att_h,context_word_att_t],3)
			context_word_att = self.linear_word_att[i](context_word_att)          #[17,17,3,128]

			#sentence_level_attention
			node_embedding_h = node_feat.unsqueeze(0).unsqueeze(2).expand_as(context_word_att)
			node_embedding_t = node_feat.unsqueeze(1).unsqueeze(2).expand_as(context_word_att)

			context_sent_att_h = self.sentence_attention[i] (sent_att_padding_matrix,context_word_att,node_embedding_h)
			context_sent_att_t = self.sentence_attention[i] (sent_att_padding_matrix,context_word_att,node_embedding_t)

			context_sent_att = torch.cat([context_sent_att_h,context_sent_att_t],2)
			context_sent_att = self.linear_sentence_att[i](context_sent_att)          #[17,17,128]

			#graphCNN
			if i<1:
				mask = torch.eq(adj_matrix,0)
				#adj_matrix_list = self.get_weighted_adj_matrix[i](node_feat,context_sent_att,mask)
				weight_adj_matrix = self.get_weighted_adj_matrix(node_feat,context_sent_att,mask)
				new_node_feat =  self.graphcnn[i] (node_feat,context_sent_att,weight_adj_matrix)
			else:
				#adj_matrix_list = self.get_weighted_adj_matrix[i](node_feat,context_sent_att)
				adj_matrix_list = self.get_adj_matrix[i-1](node_feat,context_sent_att)
				new_node_feat =  self.graphcnn[i] (node_feat,context_sent_att,adj_matrix_list)
			node_feats.append(node_feat)
			node_feat = self.config.alpha * new_node_feat + (1 - self.config.alpha) * node_feat
			#node_feat = new_node_feat + node_feat
			if is_train:
				node_feat = self.dropout(node_feat)
			#context_before_att = context_outputs[i].unsqueeze(0).unsqueeze(0).expand (node_num,node_num,max_sen_num,-1,-1)
		
		node_feats = torch.cat(node_feats,1)
		if self.use_type_feat:
			type_feats = self.ner_emb(node_type)
			node_feats_with_type = torch.cat([node_feats,type_feats],1)
		else:
			node_feats_with_type = node_feats
		
		# # print('node_feats_with_type shape',node_feats_with_type.shape)		#[node_num * 404]
		# #计算结点对的边的表示：all_pairs_shortest_path
		# entity_path_feature = torch.zeros([node_num,node_num,node_feats_with_type.size(-1)]).cuda()
		# # print('entity_path_feature',entity_path_feature.shape)
		# for i in all_pairs_shortest_path.keys():
		# 	for j in all_pairs_shortest_path[i].keys():
		# 		if i ==j:
		# 			continue
		# 		else:
		# 			path_list = all_pairs_shortest_path[i][j]
		# 			# print('path_list',path_list)
		# 			origin_path_info = torch.cat([node_feats_with_type[i].unsqueeze(0) for i in path_list],dim = 0)	#[num_num*404]
		# 			# print('origin_path_info',origin_path_info.shape)
		# 			path_len = origin_path_info.size(0)
		# 			path_info = self.rnn_path(origin_path_info.unsqueeze(0),path_len)
		# 			path_info_later = torch.mean(path_info.squeeze(0), dim = 0)
		# 			path_info_later = self.linear_path(path_info_later)
		# 			entity_path_feature[i][j] = path_info_later
		# entity_path_feature = torch.tanh(entity_path_feature)
		# entity_path_feature = self.linear_path_class(entity_path_feature)
		# # print('entity_path_feature',entity_path_feature.shape)
				# print('node_feats_with_type shape',node_feats_with_type.shape)		#[node_num * 404]
		
		# #计算节点对的边的表示 node_pairs_path_dict,all_paths_list
		# entity_path_feature = torch.zeros([node_num,node_num,node_feats_with_type.size(-1)]).cuda()
		# # print('entity_path_feature',entity_path_feature.shape)
		# path_len_tensor = torch.Tensor([(1.0/len(path)) for path in all_paths_list]).cuda()
		# # print('path_len_tensor',path_len_tensor)
		# # print('path_len_tensor',path_len_tensor.shape)
		# path_pad_list = []
		# for path in all_paths_list:
		# 	origin_path_info = torch.cat([node_feats_with_type[i].unsqueeze(0) for i in path],dim = 0)	#[path_len*404]
		# 	# print(path)
		# 	# print('origin_path_info size',origin_path_info.shape)
		# 	path_pad_list.append(origin_path_info)
		# padded_path = pad_sequence(path_pad_list)
		# # print('padded_path 1',padded_path.shape)
		# padded_path = padded_path.permute(1,0,2)
		# # print('padded_path 2',padded_path.shape)
		# # print('len all_paths_list',len(all_paths_list))	#路径数量 * 最大路径长度 * 404（dim）
		# path_after_rnn = self.rnn_path(padded_path,padded_path.size(1))
		# path_after_rnn = torch.sum(path_after_rnn,dim = 1)
		# # print('path_after_rnn',path_after_rnn.shape)

		# path_info_later = self.linear_path(path_after_rnn)
		# # print('path_info_later',path_info_later.shape)
		# # print('path_info_later',path_info_later[-4])

		# avg_path_info_later = path_info_later.t() * path_len_tensor
		# avg_path_info_later = avg_path_info_later.t()
		# # print('avg_each_path',avg_each_path.shape)
		# # print('avg_each_path',avg_each_path[-4])
		# for (i,j) in node_pairs_path_dict.keys():
		# 	# print('i',i,'j',j)
		# 	start_index = node_pairs_path_dict[(i,j)][0]
		# 	end_index = node_pairs_path_dict[(i,j)][1]
		# 	each_pair_path = torch.sum(avg_path_info_later[start_index:end_index],dim = 0)
		# 	# print('each_pair_path',each_pair_path.shape)
		# 	each_pair_path = each_pair_path / torch.tensor([end_index-start_index],dtype= torch.float32).cuda()
		# 	# print('start_index',start_index,'end_index',end_index)
		# 	entity_path_feature[i][j] == each_pair_path
		# entity_path_feature = torch.tanh(entity_path_feature)
		# entity_path_feature = self.linear_path_class(entity_path_feature)


		#计算节点对的边的表示 node_pairs_path_dict,all_paths_list
		context_sent_att = self.linear_context(context_sent_att)
		# print('context_sent_att',context_sent_att.shape)	#结点数* 结点数* 404
		# print('node_feats_with_type',node_feats_with_type.shape)	#结点数 * 404
		entity_path_feature = torch.zeros([node_num,node_num,node_feats_with_type.size(-1)]).cuda()
		# print('entity_path_feature',entity_path_feature.shape)
		path_len_tensor = torch.Tensor([(1.0/(2*len(path)-1)) for path in all_paths_list]).cuda()		#计算每条路径长度
		# print('path_len_tensor',path_len_tensor)
		# print('path_len_tensor',path_len_tensor.shape)
		path_pad_list = []
		max_path_len = 5 #抽取长度不超过2的路径
		input_list=[]
		for path in all_paths_list:
			origin_path_list = []
			# print('path_len',len(path))
			for i in range(len(path)):
				x = path[i]
				origin_path_list.append(node_feats_with_type[x].unsqueeze(0))
				if i != len(path)-1:
					y = path[i+1]
					origin_path_list.append(context_sent_att[x][y].unsqueeze(0))
			input_list.append(len(origin_path_list))
			pad_len = max_path_len - len(origin_path_list)
			for _ in range(pad_len):
				origin_path_list.append(torch.zeros(1,node_feats_with_type.size(-1)).cuda())
			# print('origin_path_list',len(origin_path_list))
			origin_path_info = torch.cat(origin_path_list,dim=0)

			path_pad_list.append(origin_path_info)
		path_pad_input = torch.stack(path_pad_list,dim=0) #844 5 404
		patdded_path = torch.nn.utils.rnn.pack_padded_sequence(path_pad_input, input_list, batch_first=True, enforce_sorted=False)
		path_after_rnn,_ = self.bilstm_path(patdded_path)
		# print('path_after_rnn',path_after_rnn)
		path_after_rnn,_ = torch.nn.utils.rnn.pad_packed_sequence(path_after_rnn,batch_first=True)
		path_after_rnn = torch.sum(path_after_rnn,dim = 1)
		path_info_later = self.linear_path(path_after_rnn)

		avg_path_info_later = path_info_later.t() * path_len_tensor
		avg_path_info_later = avg_path_info_later.t()		#path num * 404
		# print('avg_path_info_later',avg_path_info_later.shape)

		for (i,j) in node_pairs_path_dict.keys():
			start_index = node_pairs_path_dict[(i,j)][0]
			end_index = node_pairs_path_dict[(i,j)][1]
			each_pair_path = torch.sum(avg_path_info_later[start_index:end_index],dim = 0)
			# print('each_pair_path',each_pair_path.shape)
			each_pair_path = each_pair_path / torch.tensor([end_index-start_index],dtype= torch.float32).cuda()
			# print('start_index',start_index,'end_index',end_index)
			entity_path_feature[i][j] = each_pair_path
			# print('each_pair_path',each_pair_path)
		# print(entity_path_feature[0])

		# demo_list = torch.zeros(node_num,node_num,2,dtype = torch.long).cuda()
		# demo_list[:,:,1]=1
		# for (i,j) in node_pairs_path_dict.keys():
		# 	demo_list[i,j] =   torch.tensor(node_pairs_path_dict[(i,j)],dtype = torch.long).cuda()
		# print(demo_list.type())
		# print(demo_list[:,:,0].size())
		# print(avg_path_info_later[demo_list[:,:,0].view(-1):demo_list[:,:,1].view(-1)])
		# each_pair_path = torch.sum(avg_path_info_later[demo_list[:,:,0]:demo_list[:,:,1]],dim = 0)

		# demo_list = torch.zeros(node_num,node_num,avg_path_info_later.size(0),dtype = torch.float).cuda()
		# for (i,j) in node_pairs_path_dict.keys():
		# 	start_index = node_pairs_path_dict[(i,j)][0]
		# 	end_index = node_pairs_path_dict[(i,j)][1]
		# 	demo_list[i,j,start_index:end_index]=1/(end_index-start_index)
		# entity_path_feature=torch.sum(demo_list.unsqueeze(3)*avg_path_info_later.unsqueeze(0).unsqueeze(1),dim=2)
		# print(each_pair_path[0])

		entity_path_feature = self.linear_path_output(entity_path_feature)
		# print('entity_path_feature',entity_path_feature.shape)
		# entity_path_feature = torch.tanh(entity_path_feature)
		# print('entity_path_feature',entity_path_feature.shape)



		# node_feats_with_pos_h = torch.cat([node_feats_with_type.unsqueeze(1).expand(-1,node_num,-1),node_relative_pos_h],-1)
		# node_feats_with_pos_t = torch.cat([node_feats_with_type.unsqueeze(0).expand(node_num,-1,-1),node_relative_pos_t],-1)
		# print('node_feats_with_type',node_feats_with_type.shape)
		node_feats_with_pos_h = node_feats_with_type.unsqueeze(1).expand(-1,node_num,-1)
		node_feats_with_pos_t = node_feats_with_type.unsqueeze(0).expand(node_num,-1,-1)
		# print('node_feats_with_pos_h',node_feats_with_pos_h.shape)		#node_num,node_num,424
		# print('node_feats_with_pos_h[8][2]',node_feats_with_pos_t[8][2])
		# print('node_feats_with_pos_h[8][7]',node_feats_with_pos_t[6][2])
		# kkkkk

		
		entity_feature_h = torch.tanh(self.dense_layer (node_feats_with_pos_h))
		entity_feature_t = torch.tanh(self.dense_layer (node_feats_with_pos_t))
		# print('entity_feature_h',entity_feature_h)
		# print('entity_feature_t',entity_feature_t)
		# print('entity_feature_h[8][2]',entity_feature_t[8][2])
		# print('entity_feature_h[8][7]',entity_feature_t[6][2])
		# kkkkk
		# print('entity_feature_h',entity_feature_h.shape)	#node_num, node_num, 128
		# entity_feature = torch.cat([entity_feature_h,entity_feature_t],-1)		
		# cls_feature = self.linear_cls(cls_feat).unsqueeze(0).unsqueeze(0).expand(node_num,node_num,-1)
		cls_feature = self.linear_cls(cls_feat).unsqueeze(0).unsqueeze(0).expand(node_num,node_num,-1)
		# print('cls_feature',cls_feature.shape)	#node_num, node_num, 128
		# print('cls_feature',cls_feature)

		# relation_output = torch.cat([entity_feature_h,entity_feature_t,torch.abs(entity_feature_h - entity_feature_t),entity_feature_h *entity_feature_t, cls_feature, entity_path_feature ],dim =-1)
		relation_output = torch.cat((entity_feature_h,entity_feature_t,torch.abs(entity_feature_h - entity_feature_t),torch.mul(entity_feature_h ,entity_feature_t)  ),dim =-1)
		# print(relation_output.shape)
		# print('relation_output',relation_output.shape)		#node_num, node_num, 768
		relation_before_softmax_01 = torch.tanh(self.calss_linear(relation_output))
		# print('relation_before_softmax_01',relation_before_softmax_01.shape)
		relation_before_softmax_02 = self.linaer_out(relation_before_softmax_01)
		# print('relation_before_softmax_02',relation_before_softmax_02.shape)

		# relation_before_softmax_01 = self.bili_layer_01 (entity_feature_h.contiguous(),entity_feature_t.contiguous()) + self.classification_layer_01(entity_feature) + cls_feature + entity_path_feature
		# relation_before_softmax_01 =  cls_feature + entity_path_feature
		# self.calss_linear = nn.Linear(hidden_size*5,hidden_size)
		# self.linaer_out = nn.Linear(hidden_size,config.relation_num)

		# entity_feature_h_multi = torch.tanh(self.dense_layer_multi (entity_feature_h))
		# entity_feature_t_multi = torch.tanh(self.dense_layer_multi (entity_feature_t))
		# entity_feature_multi = torch.cat([entity_feature_h_multi,entity_feature_t_multi],-1)
		# relation_before_softmax_multi = self.bili_layer_multi (entity_feature_h_multi.contiguous(),entity_feature_t_multi.contiguous()) + self.classification_layer_multi(entity_feature_multi)
		#print (relation_before_softmax.size())

		return relation_before_softmax_02 #, relation_before_softmax_multi


class LockedDropout(nn.Module):             #为什么全程Dropout？理论上测试时不用dropout
	def __init__(self, dropout):
		super().__init__()
		self.dropout = dropout

	def forward(self, x):
		dropout = self.dropout
		if not self.training:
			return x
		m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)    #new用于创造一样类型的Tensor，具有相同的type和device属性，但大小不一定相同
		mask = Variable(m.div_(1 - dropout), requires_grad=False)
		mask = mask.expand_as(x)
		return mask * x

class EncoderLSTM(nn.Module):
	def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
		super().__init__()
		self.rnns = []
		for i in range(nlayers):                     #LSTM编码器，多层
			if i == 0:
				input_size_ = input_size
				output_size_ = num_units
			else:
				input_size_ = num_units if not bidir else num_units * 2                 #是否双向，改变维度
				output_size_ = num_units
			self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
		self.rnns = nn.ModuleList(self.rnns)

		self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])    #1 * 1 *hidden_dim
		self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])         #1 * 1 *hidden_dim

		self.dropout = LockedDropout(dropout)
		self.concat = concat
		self.nlayers = nlayers
		self.return_last = return_last

		# self.reset_parameters()

	def reset_parameters(self):
		for rnn in self.rnns:
			for name, p in rnn.named_parameters():
				if 'weight' in name:
					p.data.normal_(std=0.1)
				else:
					p.data.zero_()

	def get_init(self, bsz, i):              #得到初始化的两个矩阵，作为LSTM的第0个时间步的输入
		return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

	def forward(self, input, input_lengths=None):

		bsz = input.size(0)     # batchsize, padding sentence length
		output = input
		outputs = []

		for i in range(self.nlayers):                #获得初始化的LSTM第一个时间步输入
			hidden, c = self.get_init(bsz, i)

			output = self.dropout(output)            #每层开始有个dropout

			output, hidden = self.rnns[i](output, (hidden, c))     #RNN编码

			outputs.append(output)
		if self.concat:
			return torch.cat(outputs, dim=2)    #拼接
		return outputs[-1]                      #不拼接就返回最后一层

class BiAttention(nn.Module):
	def __init__(self, input_size, dropout):
		super().__init__()
		self.dropout = LockedDropout(dropout)
		self.input_linear = nn.Linear(input_size, 1, bias=False)
		self.memory_linear = nn.Linear(input_size, 1, bias=False)

		self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

	def forward(self, input, memory, mask):
		bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

		input = self.dropout(input)
		memory = self.dropout(memory)

		input_dot = self.input_linear(input)
		memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
		cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
		att = input_dot + memory_dot + cross_dot
		att = att - 1e30 * (1 - mask[:,None])

		weight_one = F.softmax(att, dim=-1)
		output_one = torch.bmm(weight_one, memory)
		weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
		output_two = torch.bmm(weight_two, input)

		return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)
