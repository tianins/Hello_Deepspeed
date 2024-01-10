# -*- coding: utf-8 -*-
import sys
sys.path.append("/data1/hqp_w/Hello_Deepspeed/")
import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader_1226 import load_data
import time
import os
import deepspeed
import torch.distributed as dist

from utils import dict_to_config_class
time_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
log_file_path = Config['log_file'] + '/' + Config['model_type'] + "_" + time_str + ".txt"
if not os.path.exists(Config['log_file']):
    os.mkdir(Config['log_file'])

# 创建一个logger对象
logger = logging.getLogger(__name__)

# 创建一个FileHandler，并设置编码为utf-8
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')

# 创建一个StreamHandler，用于将日志输出到屏幕
stream_handler = logging.StreamHandler()

# 创建一个Formatter，并设置日志格式
formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')

# 将Formatter添加到FileHandler
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# 将FileHandler添加到logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# 设置logger的日志级别
logger.setLevel(logging.INFO)


"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    global epoch
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    # train_data = load_data(config["new_train_data_path"], config)
    train_data = load_data(config["train_data_path"], config)
    logger.info("bs个数：%d" % len(train_data))
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    logger.info(config)
    config_instance = dict_to_config_class(config)
    dtype = "bf16"
    if config['deepspeed']:
        ds_config = {
            "train_micro_batch_size_per_gpu": Config["batch_size"],
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5
                }
            },
            dtype: {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu"
                }
            }
        }
        model, _, _, _ = deepspeed.initialize(model=model,
                                              model_parameters=model.parameters(),
                                              config=ds_config)
    else:
        if cuda_flag:
            logger.info("gpu可以使用，迁移模型至gpu")
            model = model.cuda()
        # 加载优化器
        optimizer = choose_optimizer(config, model)

    if config['test']:
        if config['deepspeed']:
            model.load_checkpoint(load_dir=config['load_checkpoint_dir'])
        else:
            checkpoint = torch.load(config['load_checkpoint_dir'])
            model.load_state_dict(checkpoint)
        # 加载效果测试类
        evaluator = Evaluator(config, model, logger)
        epoch = int(config['load_checkpoint_dir'].split('/')[5].split('_')[1])
        acc = evaluator.eval(epoch)
        if config['deepspeed']:
            if dist.get_rank() == 0:
                 logger.info(f"轮次: {epoch} ,acc: {acc}")
        else:
            logger.info(f"轮次: {epoch} ,acc: {acc}")
    else:
        # 加载效果测试类
        evaluator = Evaluator(config, model, logger)
        # 训练
        start_time = time.time()
        for epoch in range(config["epoch"]):
            epoch += 1
            model.train()
            logger.info("epoch %d begin" % epoch)
            train_loss = []
            for index, batch_data in enumerate(train_data):
                if cuda_flag:
                    batch_data = [d.cuda() for d in batch_data]
                if Config['deepspeed']:
                    input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
                    loss = model(input_ids, labels)
                    # Backward pass
                    model.backward(loss)
                    # Optimizer Step
                    model.step()
                else:
                    optimizer.zero_grad()
                    input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
                    loss = model(input_ids, labels)
                    loss.backward()
                    optimizer.step()

                train_loss.append(loss.item())
                if index % int(len(train_data) / 2) == 0:
                    if Config['deepspeed']:
                        if dist.get_rank() == 0:
                            print("epoch average loss: %f" % np.mean(train_loss))
                    else:
                        logger.info("batch loss %f" % loss)
            if Config['deepspeed']:
                if dist.get_rank() == 0:
                    logger.info("epoch average loss: %f" % np.mean(train_loss))
            else:
                logger.info("epoch average loss: %f" % np.mean(train_loss))
            acc = evaluator.eval(epoch)
        end_time = time.time()
        # 计算运行时间（以秒为单位）
        elapsed_time_seconds = end_time - start_time
        elapsed_time_seconds_per_e = elapsed_time_seconds/config["epoch"]
        # 转换为分钟、秒、毫秒
        elapsed_minutes = int(elapsed_time_seconds_per_e // 60)
        elapsed_seconds = int(elapsed_time_seconds_per_e % 60)
        elapsed_milliseconds = int((elapsed_time_seconds_per_e - int(elapsed_time_seconds_per_e)) * 1000)
        if Config['deepspeed']:
            if dist.get_rank() == 0:
                logger.info(f"平均训练每轮时间: {elapsed_minutes} minutes, {elapsed_seconds} seconds, {elapsed_milliseconds} milliseconds")
        else:
            logger.info(
                f"平均训练每轮时间: {elapsed_minutes} minutes, {elapsed_seconds} seconds, {elapsed_milliseconds} milliseconds")
        if not os.path.exists(config['model_path']):
            os.mkdir(config['model_path'])
        model_path = os.path.join(config["model_path"], "epoch_%d" % epoch)
        if Config['save_model']:
            if Config['deepspeed']:
                model.save_checkpoint(save_dir=model_path)
            else:
                torch.save(model.state_dict(), model_path+".pt")  #只保存模型权重
        return acc


if __name__ == "__main__":

    main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ["gated_cnn"]:
    #     Config["model_type"] = model
    #     for lr in [1e-3]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg"]:
    #                     Config["pooling_style"] = pooling_style
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)
