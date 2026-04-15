from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import uvicorn
import torch
import yaml
import numpy as np
import os
import tempfile
import time
from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from text import text_to_sequence
from pypinyin import pinyin, Style
import re
from string import punctuation

# 全局变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
vocoder = None
configs = None
args = None

# 创建FastAPI应用
app = FastAPI(
    title="FastSpeech2 TTS API",
    description="文本转语音API，基于FastSpeech2模型",
    version="1.0.0"
)

# 预加载函数
def load_model():
    global model, vocoder, configs, args
    
    # 模拟命令行参数
    class Args:
        def __init__(self):
            self.restore_step = 600000
            self.mode = "single"
            self.text = ""
            self.speaker_id = 0
            self.preprocess_config = "config/AISHELL3/preprocess.yaml"
            self.model_config = "config/AISHELL3/model.yaml"
            self.train_config = "config/AISHELL3/train.yaml"
            self.pitch_control = 1.0
            self.energy_control = 1.0
            self.duration_control = 1.0
    
    args = Args()
    
    # 读取配置
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    
    # 加载模型
    model = get_model(args, configs, device, train=False)
    
    # 加载声码器
    vocoder = get_vocoder(model_config, device)
    
    print("模型加载完成！")

# 启动时加载模型
@app.on_event("startup")
def startup_event():
    load_model()

# 读取词典
def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

# 预处理中文文本
def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

# 合成函数
def synthesize_text(text, speaker_id=0, pitch_control=1.0, energy_control=1.0, duration_control=1.0):
    global model, vocoder, configs, args
    
    preprocess_config, model_config, train_config = configs
    
    # 预处理文本
    ids = raw_texts = [text[:100]]
    speakers = np.array([speaker_id])
    texts = np.array([preprocess_mandarin(text, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    
    # 控制参数
    control_values = (pitch_control, energy_control, duration_control)
    
    # 生成语音
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # 前向传播
            output = model(
                *(batch[2:]),
                p_control=control_values[0],
                e_control=control_values[1],
                d_control=control_values[2]
            )
            
            # 创建唯一的临时目录
            import hashlib
            # 使用文本内容和时间戳创建唯一标识
            unique_id = hashlib.md5((text + str(time.time())).encode()).hexdigest()[:8]
            temp_dir = f"temp_output_{unique_id}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # 合成样本
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                temp_dir,
            )
            
            # 找到生成的wav文件
            wav_files = [f for f in os.listdir(temp_dir) if f.endswith('.wav')]
            if not wav_files:
                # 清理临时目录
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(status_code=500, detail="生成音频失败")
            
            # 返回音频文件
            wav_path = os.path.join(temp_dir, wav_files[0])
            return wav_path

# API端点
@app.get("/synthesize")
def synthesize_endpoint(
    text: str = Query(..., description="要合成的文本"),
    speaker_id: int = Query(0, description="说话人ID"),
    pitch_control: float = Query(1.0, description="音调控制，值越大音调越高"),
    energy_control: float = Query(1.0, description="能量控制，值越大音量越大"),
    duration_control: float = Query(1.0, description="时长控制，值越大语速越慢")
):
    import shutil
    temp_dir = None
    try:
        # 合成语音
        wav_path = synthesize_text(text, speaker_id, pitch_control, energy_control, duration_control)
        # 获取临时目录路径
        temp_dir = os.path.dirname(wav_path)
        
        # 返回音频文件
        return FileResponse(
            path=wav_path,
            media_type="audio/wav",
            filename=f"{text[:20]}.wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"合成失败: {str(e)}")
    finally:
        # 清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

# 健康检查端点
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "模型加载正常"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
