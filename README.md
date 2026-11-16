# 语音增强指标评测工具包
目前语音增强评测指标越来越多，本工具包的设计初衷是不同用户可以复用同一套评分代码和模型文件，从而避免不必要的部署和资源浪费。

*注：源代码请勿自行修改，有需求请联系工具包管理员*

## 🔥更新

更新通知在这里

## 用法
目前该项目文件在 `g14`，其他服务器想要使用，可以挂载；或者**联系管理员**把整个 `Evaluation` 文件夹 scp 到对应的服务器位置即可。建议存储在 `/data/hdd[x]` 盘下面。

*注：必须保证该文件夹（及其子文件夹）的所有者是管理员`nca`，避免普通用户误删改*

### Step 1: 准备元数据
1. 为测试音频创建一个 `inf.scp` 文件，每一行的格式为`[uid] [file_path]`；
2. 为参考音频创建一个 `ref.scp` 文件，每一行的格式也为 `[uid] [file_path]`;
3. 两个 `scp` 文件都放在测试音频所在的路径下（假设为 `inf_path`）；
4. 如果有需要测 WER 指标的，需要额外准备 `text` 文件，每一行的格式为 `[uid] [transcript]`；
5. 如果有需要测 CER, LID 等指标的，需要额外准备 `lang` 文件，每一行的格式为 `[uid] [language_type]`.

### Step 2: 运行脚本

每位用户只需要把主文件夹下的 `evaluate_from_path.py` 复制到自己的工作目录下，然后在该文件的开头加上以下语句：
```python3
os.chdir("/data/hdd0/Evaluation")
```

然后在终端执行以下命令即可：
```bash
python evaluate_from_path.py -I <inf_path> -D <device> -M <metric1 metric2 metric3 ...>
```

对于词错率（WER）和字错率（CER）等指标，还需要指定参考文本文件（`text`）和语音种类文件（`lang`）：
```bash
python evaluate_from_path.py -I <inf_path> -D <device> -M <metric1 metric2 metric3 ...> -T <text> -L <lang>
```


## 支持指标
注：虽然以下指标一般适用于特定的采样率，但在使用上述脚本时无需额外处理，代码内部会自动转换采样率.
| 指标 | 参数 | 说明 | 参考 |
|:----:|:---:|:-----:|:-------:|
|DNSMOS|`dnsmos`|包括了 DNSMOS P.808 和 DNSMOS P.835| × |
|NISQA| `nisqa` | 用来测大于 16k 采样率的音频的自然度 | x |
|UTMOS| `utmos` | 用来测 16k 采样率的音频的自然度，效果一般比 DNSMOS 更好 | x |
|SCOREQ| `scoreq` | 用来测 16k 采样率的音频的自然度，一般认为比 UTMOS 和 DNSMOS 更接近主观评分 | x |
|侵入式指标| `intrusive` | 包括了 PESQ, ESTOI. `calculate_intrusive_se_metrics_v2.py` 中还包含了 SDR, LSD 等 | √ |
|LPS| `phnsim` | 音素相似度. 使用 wav2vec2-lv-60-espeak-cv-ft 模型 | √ |
|SpkSim| `spksim` 或者 `spksim_wavlm` | 话者相似度，其中 `spksim` 使用 `voxcelebs12_rawnet3` 模型，`spksim_wavlm` 使用 wavlm-base-plus-sv 模型，后者的分数一般更高 | √ |
|SpeechBERTScore| `bertscore` 或者 `mbertscore`| SpeechBERTScore 指标，前者使用 HuBERT-base 测量英语数据，后者使用 mHuBERT-147 支持多语言数据 | √ |
|WER| `wer` 或者 `wer_whisper` | 词错率，前者使用 OWSM 模型，后者使用 Whisper-Large-v3 模型 | √ (text) |
|CER| `cer` 或者 `cer_whisper` | 词错率，前者使用 OWSM 模型，后者使用 Whisper-Large-v3 模型 | √ (text & lang) |
|EmoSim| `emosim` | 感情相似度，这个指标输出几乎都是0.99，没有意义 | √ |
|LID| `lidacc` | 语言准确度 | √ (lang) |

## 安装包
对于使用过程提示缺少的包，请在自己的环境下自行安装。有时候一些指标的测试需要特定版本的 `transformers` 和 `fairseq` 库，遇到类似的报错可以尝试 ChatGPT 解决。

以下对一些需要手动处理的包进行说明：
#### SCOREQ
1. 需要手动修改 scoreq 包里面的函数：`site-packages/scoreq/scoreq.py` 的 `Scoreq` 类的 `_init_onnx` 方法，第 108 行：
    ```python
    model_path = self._download_model(onnx_filename, model_url, cache_dir_name="onnx-models")
    ```
    改为：
    ```python
    model_path = "./lib/scoreq/onnx-models/adapt_nr_telephone.onnx"
    ```

#### SpeechBERTSCore
1. 依赖于 `/data/hdd0/Evaluation/lib/DiscreteSpeechMetrics-mhubert_wo_pypesq/`，安装方法：
    ```bash
    cd /data/hdd0/Evaluation/lib/DiscreteSpeechMetrics-mhubert_wo_pypesq/
    pip install .
    ```
    涉及 `build` 及 `discrete_speech_metrics.egg-info` 缓存写入的权限问题，后续补充.

2. 安装完毕后，进入安装包的初始化文件（例如`/home/nis/xiaobin.rong/anaconda3/envs/urgent2/lib/python3.10/site-packages/discrete_speech_metrics/__init__.py`），将第 4 行
    ```python
     from .speechbleu import SpeechBLEU
    ```
    注释掉.

    然后进入`/home/nis/xiaobin.rong/anaconda3/envs/urgent2/lib/python3.10/site-packages/discrete_speech_metrics/speechbertscore.py`，将第 65 行

    ```python
     self.model = AutoModel.from_pretrained('utter-project/mHuBERT-147')
    ```
    改为：
    ```python
     self.model = AutoModel.from_pretrained('./lib/DiscreteSpeechMetrics-mhubert_wo_pypesq/mHuBERT-147/')
    ```


#### WER


## 参考用例
参考用例见 `examples` 文件夹。


## 致谢
### 当前管理员
*Xiaobin Rong*

### Contributors
*Xiaobin Rong*, *Dahan Wang*

### Source
主要代码参考了 URGENT 挑战赛的[官方脚本](https://github.com/urgent-challenge/urgent2026_challenge_track1/tree/main/evaluation_metrics)


