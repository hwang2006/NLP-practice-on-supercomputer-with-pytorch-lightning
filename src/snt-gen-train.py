import torch
from ratsnlp.nlpbook.generation import GenerationTrainArguments
args = GenerationTrainArguments(
    #pretrained_model_name="skt/kogpt2-base-v2", #125M of parameters
    pretrained_model_name="skt/ko-gpt-trinity-1.2B-v0.5", #1.2B of parameters 
    downstream_corpus_name="nsmc",
    downstream_model_dir="/scratch/qualis/nlp/checkpoint-generation",
    downstream_corpus_root_dir="/scratch/qualis/nlp",
    max_seq_length=32,
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    epochs=10,
    tpu_cores=0 if torch.cuda.is_available() else 8,
    seed=7,
)

from ratsnlp import nlpbook
nlpbook.set_seed(args)

nlpbook.set_logger(args)

from Korpora import Korpora
Korpora.fetch(
    corpus_name=args.downstream_corpus_name,
    root_dir=args.downstream_corpus_root_dir,
    force_download=args.force_download,
)

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    args.pretrained_model_name,
    eos_token="</s>",
)

from ratsnlp.nlpbook.generation import NsmcCorpus, GenerationDataset
corpus = NsmcCorpus()
train_dataset = GenerationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

#from torch.utils.data import SequentialSampler
val_dataset = GenerationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="test",
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)


from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(
    args.pretrained_model_name
)


from ratsnlp.nlpbook.generation import GenerationTask
from argparse import ArgumentParser
import os

args.batch_size = 128
args.epochs = 5
args.cpu_workers = 8
args.fp16 = True

def main(cliargs):
    task = GenerationTask(model, args)
    trainer = nlpbook.get_trainer(args, cliargs)
    trainer.fit(
        task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

if __name__ ==  '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu" if torch.cuda.is_available() else "auto")
    #parser.add_argument("--accelerator", default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", default=torch.cuda.device_count() if torch.cuda.is_available() else None)
    #parser.add_argument("--strategy", default="ddp" if torch.cuda.is_available() else "auto")
    parser.add_argument("--strategy", default="ddp" if torch.cuda.is_available() else None)
    parser.add_argument("--num_nodes", default=1)

    cliargs = parser.parse_args()

    if os.getenv('SLURM_NTASKS_PER_NODE') is not None:
       cliargs.devices = os.getenv('SLURM_NTASKS_PER_NODE') # devices to be set to the slurm argument

    main(cliargs)
