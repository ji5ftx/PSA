import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)
from tqdm import tqdm

# 设置随机种子，保证可复现性
set_seed(42)


def train_next_token_predictor():
    """训练下一个token预测器，使用distilgpt2模型"""
    print("开始加载数据集...")

    # 加载数据集
    dataset = load_dataset("gabrielchua/system-prompt-leakage", split="train")
    print(f"加载了 {len(dataset)} 条系统提示词")

    # 查看数据集结构
    print("数据集结构:")
    print(f"键名: {dataset.column_names}")

    # 查看几个示例
    print("数据集示例:")
    for i in range(3):
        print(f"示例 {i + 1}:")
        for key in dataset.column_names:
            if isinstance(dataset[i][key], str):
                print(f"  {key}: {dataset[i][key][:100]}...")
            else:
                print(f"  {key}: {dataset[i][key]}")

    # 设置模型和tokenizer
    model_name = "distilgpt2"
    print(f"使用模型: {model_name}")

    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 确保tokenizer有正确的特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 数据预处理
    print("处理数据集...")

    def tokenize_function(examples):
        """将文本转换为token ID"""
        return tokenizer(
            examples["system_prompt"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    # 处理数据集
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    # 划分训练和验证集
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")

    # 设置训练参数
    output_dir = "./next-token-predictor"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,  # 尝试增大批量大小
        per_device_eval_batch_size=16,
        eval_steps=1000,  # 降低评估频率
        save_steps=2000,  # 降低保存频率
        save_total_limit=2,
        evaluation_strategy="steps",  # 根据需要可以改为更少的评估
        logging_dir="./logs",
        logging_steps=200,  # 根据需要调整日志记录频率
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        dataloader_num_workers=4  # 增加数据加载线程数
    )

    # 准备数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不使用掩码语言模型
    )

    print("加载预训练模型...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 保存最终模型
    print(f"保存模型到 {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("训练完成!")
    return output_dir


# 执行训练
if __name__ == "__main__":
    output_dir = train_next_token_predictor()
    print(f"模型已保存到: {output_dir}")