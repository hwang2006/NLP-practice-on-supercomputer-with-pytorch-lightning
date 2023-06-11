# NLP-practices-on-supercomputer-with-pytorch-lightning
Natural Language Processing (NLP) is a subfield of artificial intelligence (AI) focusing on the interaction between computers and human language. It involves the development of algorithms and models that enable computers to understand, interpret, and generate natural language in a way that is similar to how humans do. NLP encompasses a wide range of tasks including sentiment analysis, text classification, machine translation, chatbots and virtual assistants, question answering and name entity recognition. 

This repo is intended to share best practices for fine-tuning pretrained models (e.g., BERT and GPT-2) using multiple GPU nodes on a supercomputer managed by SLURM. It encompasses five examples of fine tuning : (1) Document Classification (Sentiment Analysis), (2) Sentense Pair Classification (Natural Language Inference; NLI), (3) Sequence Labeling (Named Entity Recognition), (4) Question Answering, and (5) Sentence Generation. The initial four fine tunnings rely on a BERT pretrainded model, while the Sentence Generation example uses a GPT-2 pretrained model for its fine-tunning. 

Please note that all the code in this repository is sourced from the *Ratsgo's NLPBOOK* managed by Gichang Lee. 
* You can access and run the original code on Google Colab through [Ratsgo's Tutorial Link](https://ratsgo.github.io/nlpbook/docs/tutorial_links) 
* Detailed explanations of how the five fine-tunnings work can be found in [Ratsgo's NLPBOOK](https://ratsgo.github.io/nlpbook/)

I had to make slight modifications to the original Google Colab code to enable distributed deep learning training across multiple GPU nodes on a supercomputer.  

**Contents**
* [KISTI Neuron GPU Cluster](#kisti-neuron-gpu-cluster)
* [Installing Conda](#installing-conda)
* [Installing Ratsnlp](#installing-ratsnlp)
* [Running Jupyter](#running-jupyter)
* [NLP fine-tuning code examples on Jupyter](#nlp-fine-tunings-on-jupyter) 
* [Running distributed NLP fine-tunings on SLURM](#running-distributed-nlp-fine-tunings-on-slurm)
* [Reference](#reference)

## KISTI Neuron GPU Cluster
Neuron is a KISTI GPU cluster system consisting of 65 nodes with 260 GPUs (120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>

## Installing Conda
Once logging in to Neuron, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

1. Check the Neuron system specification
```
[glogin01]$ cat /etc/*release*
CentOS Linux release 7.9.2009 (Core)
Derived from Red Hat Enterprise Linux 7.8 (Source)
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"

CentOS Linux release 7.9.2009 (Core)
CentOS Linux release 7.9.2009 (Core)
cpe:/o:centos:centos:7
```

2. Download Anaconda or Miniconda. Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices. 
```
# (option 1) Anaconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
```
```
# (option 2) Miniconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

3. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. You will install and create subsequent conda environments on your scratch directory. 
```
[glogin01]$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
[glogin01]$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here 

Miniconda3 will now be installed into this location:
/home01/qualis/miniconda3        

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home01/qualis/miniconda3] >>> /scratch/$USER/miniconda3  <======== type /scratch/$USER/miniconda3 here
PREFIX=/scratch/qualis/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
If you'd prefer that conda's base environment not be activated on startup,
   set the auto_activate_base parameter to false:

conda config --set auto_activate_base false

Thank you for installing Miniconda3!
```

4. finalize installing Miniconda with environment variables set including conda path

```
[glogin01]$ source ~/.bashrc    # set conda path and environment variables 
[glogin01]$ conda config --set auto_activate_base false
[glogin01]$ which conda
/scratch/$USER/miniconda3/condabin/conda
[glogin01]$ conda --version
conda 4.12.0
```

## Installing Ratsnlp  
Now you are ready to build a conda "nlp" virtual environment with the Ratsnlp package to be installed: 
1. load modules: 
```
module load cuda/11.7
```
2. create a new conda virtual environment and activate the environment:
```
[glogin01]$ conda create -n nlp python=3.10
[glogin01]$ conda activate nlp
```
3. install the Ratsnlp package:
```
(nlp) [glogin01]$ makdir nlp # create a new directory
(nlp) [glogin01]$ git clone https://github.com/ratsgo/ratsnlp.git # donwload ratsnlp package
Cloning into 'ratsnlp'...
remote: Enumerating objects: 928, done.
remote: Counting objects: 100% (120/120), done.
remote: Compressing objects: 100% (11/11), done.
remote: Total 928 (delta 112), reused 110 (delta 109), pack-reused 808
Receiving objects: 100% (928/928), 146.95 KiB | 0 bytes/s, done.
(nlp) [glogin01]$ ls
./  ../  ratsnlp/
(nlp) [glogin01]$ pip install --editable ratsnlp # install the ratsnlp package locally to be made modified for enabling distributed training.    
(nlp) [glogin01]$ python -c "import torch; print(torch.__version__)"
2.0.1+cu117
(nlp) [glogin01]$ python -c "import pytorch_lightning as pl; print(pl.__version__)" # make sure packages were installed. 
1.6.1
```
4. check if the pytorch lightning packages were installed:
``` 
(lightning) [glogin01]$ conda list | grep lightning
# packages in environment at /scratch/$USER/miniconda3/envs/lightning:
lightning                 2.0.2                    pypi_0    pypi
lightning-cloud           0.5.36                   pypi_0    pypi
lightning-utilities       0.8.0                    pypi_0    pypi
pytorch-lightning         2.0.2                    pypi_0    pypi
```

## Running Jupyter
[Jupyter](https://jupyter.org/) is free software, open standards, and web services for interactive computing across all programming languages. Jupyterlab is the latest web-based interactive development environment for notebooks, code, and data. The Jupyter Notebook is the original web application for creating and sharing computational documents. You will run a notebook server on a worker node (*not* on a login node), which will be accessed from the browser on your PC or labtop through SSH tunneling. 
<p align="center"><img src="https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod/assets/84169368/34a753fc-ccb7-423e-b0f3-f973b8cd7122"/>
</p>

In order to do so, you need to add the horovod-enabled virtual envrionment that you have created as a python kernel.
1. activate the horovod-enabled virtual environment:
```
[glogin01]$ conda activate lightning
```
2. install Jupyter on the virtual environment:
```
(lightning) [glogin01]$ conda install jupyter
(lightning) [glogin01]$ pip install jupyter-tensorboard
```
3. add the virtual environment as a jupyter kernel:
```
(lightning) [glogin01]$ pip install ipykernel 
(lightning) [glogin01]$ python -m ipykernel install --user --name lightning
```
4. check the list of kernels currently installed:
```
(lightning) [glogin01]$ jupyter kernelspec list
Available kernels:
python3       /home01/$USER/.local/share/jupyter/kernels/python3
horovod       /home01/$USER/.local/share/jupyter/kernels/lightning
```
5. launch a jupyter notebook server on a worker node 
- to deactivate the virtual environment
```
(horovod) [glogin01]$ conda deactivate
```
- to create a batch script for launching a jupyter notebook server: 
```
[glogin01]$ cat jupyter_run.sh
#!/bin/bash
#SBATCH --comment=tensorflow
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
#SBATCH --time=8:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=4     # number of cpus per task

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the node name and port
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

echo "load module-environment"
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1

echo "execute jupyter"
source ~/.bashrc
conda activate lightning
cd /scratch/$USER     # the root/work directory of the jupyter lab/notebook to be launched
jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER} #jupyter token: your user name
echo "end of the job"
```
- to launch a jupyter notebook server 
```
[glogin01]$ sbatch jupyter_run.sh
Submitted batch job XXXXXX
```
- to check if a jupyter notebook server is running
```
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX    amd_a100nv_8 jupyter_    $USER  RUNNING       0:02   8:00:00      1 gpu30
[glogin01]$ cat slurm-XXXXXX.out
.
.
[I 2023-02-14 08:30:04.790 ServerApp] Jupyter Server 1.23.4 is running at:
[I 2023-02-14 08:30:04.790 ServerApp] http://gpu##:#####/lab?token=...
.
.
```
- to check the SSH tunneling information generated by the jupyter_run.sh script 
```
[glogin01]$ cat port_forwarding_command
ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
```
6. open a SSH client (e.g., Putty, PowerShell, Command Prompt, etc) on your PC or laptop and log in to Neuron just by copying and pasting the port_forwarding_command:
```
C:\Users\hwang>ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
Password(OTP):
Password:
```
7. open a web browser on your PC or laptop to access the jupyter server
```
URL Address: localhost:8888e
Password or token: $USER    # your account ID on Neuron
```
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218938419-f38c356b-e682-4b1c-9add-6cfc29d53425.png"/></p> 

## NLP fine-tunings on Jupyter
Now, you are ready to run fine-tuning code examples on a jupyter notebook. Please refer to the [notebooks](https://github.com/hwang2006/NLP-practice-on-supercomputer-with-pytorch-lightning/tree/main/notebooks) directory.
* [Sentiment Analysis](https://nbviewer.org/github/hwang2006/NLP-practice-on-supercomputer-with-pytorch-lightning/blob/main/notebooks/doc-cls-train.ipynb)
* [Natural Language Inference](https://nbviewer.org/github/hwang2006/NLP-practice-on-supercomputer-with-pytorch-lightning/blob/main/notebooks/pair-cls-train.ipynb)
* [Named Entiry Recognition](https://nbviewer.org/github/hwang2006/NLP-practice-on-supercomputer-with-pytorch-lightning/blob/main/notebooks/ner-train.ipynb)
* [Question Answering](https://nbviewer.org/github/hwang2006/NLP-practice-on-supercomputer-with-pytorch-lightning/blob/main/notebooks/QA-train.ipynb)
* [Sentence Generation](https://nbviewer.org/github/hwang2006/NLP-practice-on-supercomputer-with-pytorch-lightning/blob/main/notebooks/snt-gen-train.ipynb)

## Running distributed NLP fine-tunings on SLURM
Now, you are ready to run the NLP fine-tuning examples on multiple GPU nodes. You need to tweak the Ratsnlp package a bit in order to be able to conduct distributed fine-tunning practices on a SLURM cluster. Yon need to modify the "get_trainer()" method in the trainer.py to enable the execution of distributed training across multiple GPU nodes. That's it!!
```
[glogin01]$ ls
./  ../  ratsnlp/
[glogin01]$ cat ./ratsnlp/ratsnlp/nlpbook/trainer.py
import os
import torch
from pytorch_lightning import Trainer
#from lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def get_trainer(args, cliargs, return_trainer_only=True):
    ckpt_path = os.path.abspath(args.downstream_model_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=args.save_top_k,
        monitor=args.monitor.split()[1],
        mode=args.monitor.split()[0],
        #filename='{epoch}-{val_loss:.2f}',
        filename='{epoch}-{val_loss:.3f}',
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        callbacks=[checkpoint_callback],
        default_root_dir=ckpt_path,
        # For GPU Setup
        #deterministic=torch.cuda.is_available() and args.seed is not None,
        #gpus=torch.cuda.device_count() if torch.cuda.is_available() else None,
        accelerator=cliargs.accelerator,
        devices=cliargs.devices,
        #strategy="ddp" if torch.cuda.is_available() else "auto",
        strategy=cliargs.strategy,
        num_nodes = cliargs.num_nodes,
        precision=16 if args.fp16 else 32,
        # For TPU Setup
        tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )
    if return_trainer_only:
        return trainer
    else:
        return checkpoint_callback, trainer
```

Here's the orginal get_trainer method before modification, where you may notice that the get_trainer method has been modified to accept command line arguments that has to do with distributed training including "strategy" and "num_nodes" as for the Trainer parameters of the underlying Pytorch Lightning Framework. 
```
[glogin01]$ cat ./ratsnlp/ratsnlp/nlpbook/trainer.org.py
import os 
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def get_trainer(args, return_trainer_only=True):
    ckpt_path = os.path.abspath(args.downstream_model_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=args.save_top_k,
        monitor=args.monitor.split()[1],
        mode=args.monitor.split()[0],
        filename='{epoch}-{val_loss:.2f}',
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        callbacks=[checkpoint_callback],
        default_root_dir=ckpt_path,
        # For GPU Setup
        deterministic=torch.cuda.is_available() and args.seed is not None,
        gpus=torch.cuda.device_count() if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32,
        # For TPU Setup
        tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )
    if return_trainer_only:
        return trainer
    else:
        return checkpoint_callback, trainer
```
Now, you will be able to carry out distributed fine-tunning practices on a supercomputer. All the fine-tunning code is writted based on the one
that ran successfully on Jupyter. All the source code is avaiable in the [src](https://github.com/hwang2006/NLP-practice-on-supercomputer-with-pytorch-lightning/tree/main/src) directory.  

1. request allocation of available GPU-nodes:
```
[glogin01]$ salloc --partition=amd_a100nv_8 -J debug --nodes=2 --time=8:00:00 --gres=gpu:4 --comment=python
salloc: Granted job allocation 154173
salloc: Waiting for resource configuration
salloc: Nodes gpu[32-33] are ready for job
```
2. load modules and activate the nlp conda environment:
```
[gpu32]$ module load cuda/11.7
[gpu32]$ $ conda activate nlp 
(nlp) [gpu32]$
```
3. run a fine-tunning code:

Here is one of the fine-tuning examples in the [src](https://github.com/hwang2006/NLP-practice-on-supercomputer-with-pytorch-lightning/tree/main/src) directory, where you might notice that you may want to switch its pretrained model with an appropriate pretrained model for your purpose. Please refer to the [customization section](https://ratsgo.github.io/nlpbook/docs/doc_cls/detail/) of Ratsgo's NLPBOOK for getting an idea of how to customize the fine-tuning code examples for your own task. 
```
[glogin01]$ cat NLP-practice-on-supercomputer-with-pytorch-lightning/src/doc-cls-train.py
import torch
from ratsnlp.nlpbook.classification import ClassificationTrainArguments
args = ClassificationTrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    #pretrained_model_name="beomi/kcbert-large",
    downstream_corpus_name="nsmc",
    downstream_model_dir="/scratch/qualis/nlp/checkpoint-doccls",
    downstream_corpus_root_dir="/scratch/qualis/nlp",
    #batch_size=32 if torch.cuda.is_available() else 4,
    batch_size=128 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    max_seq_length=128,
    #cpu_workers=8,
    #epochs=1,
    epochs=3,
    #fp16=True,
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
)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset
corpus = NsmcCorpus()
train_dataset = ClassificationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)

from torch.utils.data import DataLoader, RandomSampler
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

from torch.utils.data import SequentialSampler
val_dataset = ClassificationDataset(
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

from transformers import BertConfig, BertForSequenceClassification
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=corpus.num_labels,
)
model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)

from ratsnlp.nlpbook.classification import ClassificationTask
from argparse import ArgumentParser
import os

args.batch_size = 128
args.epochs = 3
args.cpu_workers = 8
args.fp16 = True

def main(cliargs):
        task = ClassificationTask(model, args)
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
```
- to run on the two nodes with 4 GPUs each. Pytorch Lightning complains and exits with some runtime error messages when using "srun" with the -n or --ntasks options, so you need to use --ntasks-per-node instead.
```
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/doc-cls-train.py --num-nodes 2
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/pair-cls-train.py --num-nodes 2
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/ner_train.py --num-nodes 2
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/QA_train.py --num-nodes 2
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/snt-gen-train.py --num-nodes 2
```
- to run on the two nodes with 2 GPUs each
```
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/doc-cls-train.py --num_nodes 2 --devices 2
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/pair-cls-train.py --num_nodes 2 --devices 2
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/ner_train.py --num-nodes 2 --devices 2
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=2 NLP-practice-on-supercomputer-with-pytorch-lightning/src/QA_train.py --num-nodes 2 --devices 2
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/snt-gen-train.py --devices 2
```
- to run on the two nodes with 1 GPU each
```
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=1 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/doc-cls-train.py --num_nodes 2 --devices 1
(nlp) [gpu32]$ srun -N 2 --ntasks-per-node=1 NLP-practice-on-supercomputer-with-pytorch-lightning/src/pair-cls-train.py --num_nodes 2 --devices 1
```
- to run one node with 4 GPUs
```
(nlp) [gpu32]$ python NLP-practice-on-supercomputer-with-pytorch-lightning/src/doc-cls-train.py
(nlp) [gpu32]$ python NLP-practice-on-supercomputer-with-pytorch-lightning/src/pair-cls-train.py
(nlp) [gpu32]$ srun -N 1 --ntasks-per-node=4 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/doc-cls-train.py
(nlp) [gpu32]$ srun -N 1 --ntasks-per-node=4 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/pair-cls-train.py
```
- to run one node with 2 GPUs
```
(nlp) [gpu32]$ python NLP-practice-on-supercomputer-with-pytorch-lightning/src/doc-cls-train.py --devices 2
(nlp) [gpu32]$ python NLP-practice-on-supercomputer-with-pytorch-lightning/src/pair-cls-train.py --devices 2
(nlp) [gpu32]$ srun -N 1 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/doc-cls-train.py
(nlp) [gpu32]$ srun -N 1 --ntasks-per-node=2 python NLP-practice-on-supercomputer-with-pytorch-lightning/src/pair-cls-train.py
```
Now, you are ready to submit and run a fine-tuning batch job.
1. edit a batch job script running on 2 nodes with 8 GPUs each:
```
[glogin01]$ cat pl_finetunnig_batch.sh
#!/bin/sh
#SBATCH -J pytorch_lightning # job name
#SBATCH --time=24:00:00 # walltime
#SBATCH --comment=pytorch # application name
#SBATCH -p amd_a100nv_8 # partition name (queue or class)
#SBATCH --nodes=2 # the number of nodes
#SBATCH --ntasks-per-node=8 # number of tasks per node
#SBATCH --gres=gpu:8 # number of GPUs per node
#SBATCH --cpus-per-task=4 # number of cpus per task
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load cuda/11.7
source ~/.bashrc
conda activate nlp

# The num_nodes argument should be specified to be the same number as in the #SBATCH --nodes=xxx
srun python NLP-practice-on-supercomputer-with-pytorch-lightning/src/doc-cls-train.py --num_nodes 2
```
2. submit and execute the batch job:
```
[glogin01]$ sbatch pl_finetuning_batch.sh
Submitted batch job 169608
```
3. check & monitor the batch job status:
```
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            169608    amd_a100nv_8   python   qualis  PENDING       0:00 1-00:00:00      2 (Resources)
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            169616    amd_a100nv_8   python   qualis  RUNNING       1:01 1-00:00:00      2 gpu[32,34]
```
4. check the standard output & error files of the batch job:
```
[glogin01]$ cat pytorch_lightning_169608.out
[glogin01]$ cat pytorch_lightning_169608.err
```
5. For some reason, you may want to stop or kill the batch job.
```
[glogin01]$ scancel 169608
```
## Reference
* [Ratsnlp GitHub](https://github.com/ratsgo/ratsnlp)
