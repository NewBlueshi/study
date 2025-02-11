import onnxruntime as ort
import numpy as np
from onnxruntime import InferenceSession
from transformers import MarianTokenizer
import logging
from typing import Tuple, List, Dict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_models(model_dir: str) -> tuple[InferenceSession, InferenceSession]:
    """
    加载ONNX模型
    :param model_dir: 模型目录路径
    :return: 编码器会话, 解码器会话, 带历史记录的解码器会话
    """
    try:
        encoder_session = ort.InferenceSession(f"{model_dir}/encoder_model.onnx", providers=['CPUExecutionProvider'])
        decoder_session = ort.InferenceSession(f"{model_dir}/decoder_model_merged.onnx",
                                               providers=['CPUExecutionProvider'])
        return encoder_session, decoder_session
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise


def preprocess_input(text: str, tokenizer: MarianTokenizer) -> Dict[str, np.ndarray]:
    """
    预处理输入文本
    :param text: 输入文本
    :param tokenizer: 分词器
    :return: 编码器输入字典
    """
    if not text or len(text) > 512:  # 限制最大长度
        raise ValueError("输入文本长度必须在1到512个字符之间")

    # 添加语言标记
    text = f">>cmn_Hans<< {text}"
    inputs = tokenizer(text, return_tensors="np")
    return {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }


def translate_text(
        input_text: str,
        tokenizer: MarianTokenizer,
        encoder_session: ort.InferenceSession,
        decoder_session: ort.InferenceSession,
        max_length: int = 64
) -> str:
    """
    翻译文本
    :param input_text: 输入文本
    :param tokenizer: 分词器
    :param encoder_session: 编码器会话
    :param decoder_session: 解码器会话
    :param max_length: 最大输出长度
    :return: 翻译后的文本
    """
    try:
        # 预处理输入
        encoder_inputs = preprocess_input(input_text, tokenizer)
        # 编码器推理
        encoder_outputs = encoder_session.run(None, encoder_inputs)
        encoder_hidden_states = encoder_outputs[0]

        # 初始化解码器
        decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
        output_ids = []
        outputs = []
        output_encode_past = []
        decoder_inputs = {
            "input_ids": decoder_input_ids,
            "encoder_attention_mask": encoder_inputs["attention_mask"],
            "use_cache_branch": np.array([True], dtype=np.bool_)
        }

        # 解码循环
        for index in range(max_length):
            print("======> decode run : ", index)
            # 准备解码器输入
            if index == 0:
                for i in range(6):  # 模型有6层 Optional[Tuple[Tuple[torch.FloatTensor]]]
                    decoder_inputs[f"past_key_values.{i}.decoder.key"] = np.zeros((1, 8, 1, 64), dtype=np.float32)
                    decoder_inputs[f"past_key_values.{i}.decoder.value"] = np.zeros((1, 8, 1, 64), dtype=np.float32)
                    decoder_inputs[f"past_key_values.{i}.encoder.key"] = np.zeros((1, 8, 8, 64), dtype=np.float32)
                    decoder_inputs[f"past_key_values.{i}.encoder.value"] = np.zeros((1, 8, 8, 64), dtype=np.float32)
                decoder_inputs["encoder_hidden_states"] = encoder_hidden_states
                decoder_inputs["use_cache_branch"] = np.array([False], dtype=np.bool_)
                outputs = decoder_session.run(None, decoder_inputs)
                output_encode_past = outputs
            else:
                # 添加历史记录
                for i in range(6):  # 模型有6层
                    decoder_inputs[f"past_key_values.{i}.decoder.key"] = outputs[1 + 4 * i]
                    decoder_inputs[f"past_key_values.{i}.decoder.value"] = outputs[2 + 4 * i]
                    decoder_inputs[f"past_key_values.{i}.encoder.key"] = output_encode_past[3 + 4 * i]
                    decoder_inputs[f"past_key_values.{i}.encoder.value"] = output_encode_past[4 + 4 * i]

                decoder_inputs["use_cache_branch"] = np.array([True], dtype=np.bool_)

                outputs = decoder_session.run(None, decoder_inputs)

            # 获取下一个token
            logits = outputs[0]
            next_token_id = np.argmax(logits[:, -1, :], axis=-1)
            output_ids.append(next_token_id.item())

            # 如果遇到结束符则停止
            if next_token_id == tokenizer.eos_token_id:
                break

            # 更新解码器输入
            decoder_input_ids = next_token_id.reshape(1, 1)
            decoder_inputs["input_ids"] = decoder_input_ids

        # 解码输出文本
        translated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return translated_text
    except Exception as e:
        logger.error(f"翻译过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # 初始化
        # model_dir = "opus-mt-en-zh"
        model_dir = "."
        tokenizer = MarianTokenizer.from_pretrained(model_dir)
        encoder_session, decoder_session = load_models(model_dir)

        # 示例输入
        input_text = "This is a test sentence."

        # 翻译
        translated_text = translate_text(
            input_text,
            tokenizer,
            encoder_session,
            decoder_session
        )

        print(f"Translated text: {translated_text}")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
