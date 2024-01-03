# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import transformers
from config import Config
from BingBertGlue.pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import dict_to_config_class

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.ds_bert = False
        self.ds_clf_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = transformers.BertModel.from_pretrained(config["pretrain_model_path"])
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "ds_bert":
            bert_base_model_config = {
                "vocab_size_or_config_json_file": 119547,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 512,
                "type_vocab_size": 2,
                "initializer_range": 0.02
            }
            if Config["progressive_layer_drop"]:
                print("BertBaseConfigPreLnLayerDrop")
                # 使用nvidia剪枝模型
                from BingBertGlue.nvidia.modelingpreln_layerdrop import BertModel, BertForSequenceClassification, BertConfig
            elif Config["preln"]:
                from BingBertGlue.nvidia.modelingpreln import BertModel,BertForSequenceClassification, BertConfig, BertLayer
            else:
                from BingBertGlue.nvidia.modeling import BertModel,BertForSequenceClassification, BertConfig, BertLayer
            config_instance = dict_to_config_class(config)
            bert_config = BertConfig(**bert_base_model_config)
            # if config_instance.local_rank == -1 or config_instance.no_cuda:
            #     device = torch.device(
            #         "cuda" if torch.cuda.is_available() and not config_instance.no_cuda else "cpu")
            #     n_gpu = torch.cuda.device_count()
            # else:
            #     torch.cuda.set_device(config_instance.local_rank)
            #     device = torch.device("cuda", config_instance.local_rank)
            #     n_gpu = 1
            #     # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            #     torch.distributed.init_process_group(backend='nccl')
            # print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            #     device, n_gpu, bool(config_instance.local_rank != -1), config_instance.fp16))
            #

            tokenizer = BertTokenizer.from_pretrained(
                config["bert_model"], do_lower_case=config["do_lower_case"])
            bert_config.vocab_size = len(tokenizer.vocab)
            # Padding for divisibility by 8
            if bert_config.vocab_size % 8 != 0:
                bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)
            # 不清楚这里的args里面都哪些有用哪些没用，
            if config['use_classification_model']:
                self.encoder = BertForSequenceClassification(config_instance, bert_config, num_labels=class_num)
                self.ds_clf_bert = True
            else:
                self.encoder = BertModel(bert_config, config_instance)
                self.ds_bert = True
            # if config_instance.fp16:
            #     self.encoder.half()
            # self.encoder.to(device)
            # if config_instance.local_rank != -1:
            #     try:
            #         if config_instance.deepscale:
            #             print("Enabling DeepScale")
            #             from deepscale.distributed_apex import DistributedDataParallel as DDP
            #         else:
            #             from apex.parallel import DistributedDataParallel as DDP
            #     except ImportError:
            #         raise ImportError(
            #             "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            #
            #     self.encoder = DDP(self.encoder)
            # elif n_gpu > 1:
            #     self.encoder = torch.nn.DataParallel(self.encoder)



            # self.encoder = BertModel.from_pretrained(config["pretrain_model_path"])
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # 直接使用分类模型
        if self.ds_clf_bert:
            predict = self.encoder(x, labels=target)  # input shape:(batch_size, input_dim)
            return predict
        else:
            # x int64，这里的x中的每一个元素表示，一个字符对应在词表中的序号 bs*len
            # 之后经过embedding层，将其转变为float类型的词嵌入，增加一个维度，再消去sen_len维度，这样才能送入线性层
            # （线性层的权重和偏置定义为浮点数类型，因为这可以更好地处理连续值的情况，不能直接使用int类型）
            if self.use_bert:  # bert返回的结果是 (sequence_output, pooler_output)
                x = self.encoder(x)
                x = x.last_hidden_state  # bs,len,hs
            elif self.ds_bert:  # 使用nvidia模型
                all_encoder_layers, pooled_output = self.encoder(x)  # all_encoder_layers 12层的输出结果，取最后一维
                x = all_encoder_layers[-1]  # bs,len,hs
            else:
                x = self.embedding(x)  # embedding(vocab_size,hs) 输入x:(batch_size, sen_len) 输出x(bs,sen_len,hs),增加hs维度
                x = self.encoder(x)  # input shape:(batch_size, sen_len, input_dim) encoder(hs,hs) 输出 （hs,hs）
            if isinstance(x, tuple):  # RNN类的模型会同时返回隐单元向量，我们只取序列结果
                x = x[0]
            # 可以采用pooling的方式得到句向量
            if self.pooling_style == "max":
                self.pooling_layer = nn.MaxPool1d(x.shape[1])
            else:
                self.pooling_layer = nn.AvgPool1d(x.shape[1])
            x = self.pooling_layer(x.transpose(1, 2)).squeeze()  # input shape:(batch_size, sen_len, input_dim)

            # 也可以直接使用序列最后一个位置的向量
            # x = x[:, -1, :]
            predict = self.classify(x)  # input shape:(batch_size, input_dim)
            if target is not None:
                return self.loss(predict, target.squeeze())
            else:
                return predict


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):  # x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        # ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        # 仿照bert的transformer模型结构，将self-attention替换为gcnn
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  # 通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  # 之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  # 一层线性
            l1 = torch.relu(l1)  # 在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1)  # 二层线性
            x = self.bn_after_ff[i](x + l2)  # 残差后过bn
        return x


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x


class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config["pretrain_model_path"])
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x


class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config["pretrain_model_path"])
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x


class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config["pretrain_model_path"])
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        layer_states = self.bert(x)[2]
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states


# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    # Config["class_num"] = 3
    # Config["vocab_size"] = 20
    # Config["max_length"] = 5
    Config["model_type"] = "bert"
    model = transformers.BertModel.from_pretrained(Config["pretrain_model_path"])
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print(x[2], type(x[2]), len(x[2]))

    # model = TorchModel(Config)
    # label = torch.LongTensor([1,2])
    # print(model(x, label))
