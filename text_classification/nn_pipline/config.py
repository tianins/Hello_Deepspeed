# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    # Bert pre-trained model selected in the list: bert-base-uncased, "
    # "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
    # "bert-base-multilingual-cased, bert-base-chinese.
    "deepspeed":True,
    "test":True,
    "save_model": True,
    "load_checkpoint_dir":'/data1/hqp_w/Hello_Deepspeed/output/epoch_2/', # '/data1/hqp_w/Hello_Deepspeed/output/epoch_5/global_step500/'
    "bert_model":"bert-base-chinese",
    "use_classification_model":False,
    "max_seq_length":128,
    "do_lower_case":True,
    "no_cuda":False,
    "local_rank":1,
    "gradient_accumulation_steps":1,
    "fp16":True,
    "loss_scale":0,
    "model_file":0,
    "random":False,
    "focal":False,
    "gamma":0.5,
    "deepspeed_transformer_kernel":False,
    "progressive_layer_drop":False,
    "preln":False,


    
    
    "log_file": "log_file",
    "class_num": 2,
    "model_path": "output",
    # "new_train_data_path": "../data/Sentiment_classification_data_processing/new_train_data.json",
    "train_data_path": "/data1/hqp_w/Hello_Deepspeed/text_classification/data/Sentiment_classification_data_processing/new_train_data.json",
    "valid_data_path": "/data1/hqp_w/Hello_Deepspeed/text_classification/data/Sentiment_classification_data_processing/test_data.json",
    "vocab_path": "/data1/hqp_w/Hello_Deepspeed/text_classification/nn_pipline/chars.txt",
    "model_type": "bert",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 4,
    "batch_size": 64,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path": r"/data1/hqp_w/pre_train_model/models--bert-base-chinese/",
    "seed": 987
}
