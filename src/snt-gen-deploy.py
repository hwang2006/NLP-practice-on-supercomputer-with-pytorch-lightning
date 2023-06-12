from ratsnlp.nlpbook.generation import GenerationDeployArguments
args = GenerationDeployArguments(
    pretrained_model_name="skt/kogpt2-base-v2",
    #pretrained_model_name="skt/ko-gpt-trinity-1.2B-v0.5",
    downstream_model_dir="/scratch/qualis/nlp/checkpoint-generation",
)

print(args.downstream_model_checkpoint_fpath)

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    args.pretrained_model_name,
    eos_token="</s>",
)

import torch
from transformers import GPT2Config, GPT2LMHeadModel
#if args.downstream_model_checkpoint_path is None:
if args.downstream_model_checkpoint_fpath is None:
    model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model_name,
    )
else:
    #from google.colab import drive
    #drive.mount('/gdrive', force_remount=True)    
    pretrained_model_config = GPT2Config.from_pretrained(
        args.pretrained_model_name,
    )
    model = GPT2LMHeadModel(pretrained_model_config)
    fine_tuned_model_ckpt = torch.load(
        #args.downstream_model_checkpoint_path,
        args.downstream_model_checkpoint_fpath,
        map_location=torch.device("cpu"),
    )
    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
    #model.load_state_dict({k.replace("transformer.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})

model.eval()

def inference_fn(
        prompt,
        min_length=10,
        max_length=20,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        temperature=1.0,
):
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=True,
                top_p=float(top_p),
                top_k=int(top_k),
                min_length=int(min_length),
                max_length=int(max_length),
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                temperature=float(temperature),
           )
        generated_sentence = tokenizer.decode([el.item() for el in generated_ids[0]])
    except:
        generated_sentence = """처리 중 오류가 발생했습니다. <br>
            변수의 입력 범위를 확인하세요. <br><br> 
            min_length: 1 이상의 정수 <br>
            max_length: 1 이상의 정수 <br>
            top-p: 0 이상 1 이하의 실수 <br>
            top-k: 1 이상의 정수 <br>
            repetition_penalty: 1 이상의 실수 <br>
            no_repeat_ngram_size: 1 이상의 정수 <br>
            temperature: 0 이상의 실수
            """
    return {
        'result': generated_sentence,
    }

from ratsnlp.nlpbook.generation import get_web_service_app
app = get_web_service_app(inference_fn, is_colab=False)
#app.run()

if __name__ == "__main__":
    app.run()
