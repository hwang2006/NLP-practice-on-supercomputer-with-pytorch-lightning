import torch
from ratsnlp.nlpbook.ner import NERTrainArguments
args = NERTrainArguments(
    #pretrained_model_name="beomi/kcbert-base",
    pretrained_model_name="beomi/kcbert-large",
    downstream_corpus_name="ner",
    downstream_model_dir="/scratch/qualis/nlp/checkpoint-ner",
    downstream_corpus_root_dir="/scratch/qualis/nlp",
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    max_seq_length=64,
    #epochs=3,
    epochs=5,
    tpu_cores=0 if torch.cuda.is_available() else 8,
    seed=7,
)

from ratsnlp import nlpbook
nlpbook.set_seed(args)

nlpbook.download_downstream_dataset(args)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

from ratsnlp.nlpbook.ner import NERCorpus, NERDataset
corpus = NERCorpus(args)
train_dataset = NERDataset(
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

val_dataset = NERDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="val",
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

from transformers import BertConfig, BertForTokenClassification
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=corpus.num_labels,
)
model = BertForTokenClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)

#from ratsnlp.nlpbook.classification import ClassificationTask
from ratsnlp.nlpbook.ner import NERTask
from argparse import ArgumentParser
import os

args.batch_size = 128
args.epochs = 3
args.cpu_workers = 8
args.fp16 = True

def main(cliargs):
    task = NERTask(model, args)
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
