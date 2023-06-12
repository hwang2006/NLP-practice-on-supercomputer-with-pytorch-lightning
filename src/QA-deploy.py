from ratsnlp.nlpbook.qa import QADeployArguments
args = QADeployArguments(
    #pretrained_model_name="beomi/kcbert-base",
    pretrained_model_name="beomi/kcbert-large",
    downstream_model_dir="/scratch/qualis/nlp/checkpoint-qa",
    max_seq_length=128,
    max_query_length=32,
)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

import torch
fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cpu")
)

from transformers import BertConfig
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
)

from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering(pretrained_model_config)

model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})

model.eval()

def inference_fn(question, context):
    if question and context:
        truncated_query = tokenizer.encode(
            question,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_query_length
       )
        inputs = tokenizer.encode_plus(
            text=truncated_query,
            text_pair=context,
            truncation="only_second",
            padding="max_length",
            max_length=args.max_seq_length,
            return_token_type_ids=True,
        )
        with torch.no_grad():
            outputs = model(**{k: torch.tensor([v]) for k, v in inputs.items()})
            start_pred = outputs.start_logits.argmax(dim=-1).item()
            end_pred = outputs.end_logits.argmax(dim=-1).item()
            pred_text = tokenizer.decode(inputs['input_ids'][start_pred:end_pred+1])
    else:
        pred_text = ""
    return {
        'question': question,
        'context': context,
        'answer': pred_text,
    }

from ratsnlp.nlpbook.qa import get_web_service_app
app = get_web_service_app(inference_fn, is_colab=False)
#app.run()

if __name__ == "__main__":
    app.run()
