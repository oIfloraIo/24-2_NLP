# nlp-homework.ipynb
meta-llama/Llama-3.2-1B-Instruct 모델을 주어진 데이터셋을 통해 파인튜닝하여 요청받은 주문에 대한 대응이 가능한 모델로 확장

## Q3. Upload the fine-tuned LoRA adapter to the Hugging Face Hub
**업로드한 모델 경로** : 

https://huggingface.co/shlee0/llama-3.2.1-finetuning-assign

**모델 finetuning 진행 시 지표** (wandb)
![image](https://github.com/user-attachments/assets/5f2c79ee-e5c6-428e-b2ee-c4262c2469fc)
https://wandb.ai/floralee_782/huggingface/runs/sp1b614l?nw=nwuserfloralee782


##### train/learning_rate
MAX : llama-3.2-1b-fine-tuning train/learning_rate	**0.0001**

MIN : llama-3.2-1b-fine-tuning train/learning_rate	**0**


##### train/grad_norm
llama-3.2-1b-fine-tuning train/grad_norm	**1.4947929382324219**

##### train/loss
llama-3.2-1b-fine-tuning train/loss	**0.5278**

## Q2. Use only 2,800 samples from the dataset for fine-tuning
### 학습 모델 평가 (200개의 validation data)
![image](https://github.com/user-attachments/assets/e641eb12-19db-468c-9cac-3f9ed2cd5ac4)
```
full_dataset = Dataset.from_json(path_or_paths=dataset_name)

train_dataset = full_dataset.select(range(2800))
valid_dataset = full_dataset.select(range(2800, 3000))
```

## Q5. Measure the BLEU score for each output in the validation set and the generated output
**sacre-bleu** 를 통한 BLEU score 확인
**BLEU score: 87.2813115602023**

### 학습 시 추가 설정한 인자값

> **학습 시 설정한 peft**
```
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
```

> **학습 시 활용 PROMPT 조건 추가**
```
def function_prepare_sample_text(tokenizer, for_train=True):
    def _prepare_sample_text(example):
        system_prompt = "너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 음식명, 옵션명, 수량을 추출한다."
        user_input = f"### 주문 문장:\n{example['input']}"
        messages = [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"{user_input}"},
        ]
        if for_train:
            messages.append({"role": "assistant", "content": f"{example['output']}"})

        text = ""
        for message in messages:
            if message['role'] == 'system':
                text += f"<s>[SYSTEM]\n{message['content']}\n[/SYSTEM]"
            elif message['role'] == 'user':
                text += f"\n[USER]\n{message['content']}\n[/USER]"
            elif message['role'] == 'assistant':
                text += f"\n[ASSISTANT]\n{message['content']}\n[/ASSISTANT]</s>"
        return text

    return _prepare_sample_text
```

