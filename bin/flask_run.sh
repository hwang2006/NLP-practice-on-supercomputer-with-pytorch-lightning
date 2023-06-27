#!/bin/bash 
#SBATCH --comment=tensorflow
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
##SBATCH --partition=cas_v100nv_8
##SBATCH --partition=cas_v100_4
#SBATCH --time=1:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes 
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=4     # number of cpus per task

#removing the old port forwading
if [ -e flask_port_forwarding_command ]
then
  rm flask_port_forwarding_command
fi

#getting the port and node name
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU 

echo "ssh -L localhost:5000:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > flask_port_forwarding_command
echo "ssh -L localhost:5000:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > flask_port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

echo "load module-environment"
module load cuda/11.7

echo "execute flask"
source ~/.bashrc
conda activate nlp
#conda activate tf-nlp
#conda activate hvd
cd /scratch/qualis/nlp  
flask --app NLP-practices-on-supercomputer-with-pytorch-lightning/src/doc-cls-deploy.py run --host=0.0.0.0 --port=${PORT_JU} --debug
#flask --app doc-cls-deploy.py run --host=0.0.0.0 --port=${PORT_JU} --debug
#flask --app pair-cls-deploy.py run --host=0.0.0.0 --port=${PORT_JU} --debug
#flask --app ner-deploy.py run --host=0.0.0.0 --port=${PORT_JU} --debug
#flask --app QA-deploy.py run --host=0.0.0.0 --port=${PORT_JU} --debug
#flask --app snt-gen-deploy.py run --host=0.0.0.0 --port=${PORT_JU} --debug

#if [ -f "$1" ]
#then
#	flask --app $1 run --host=0.0.0.0 --port=${PORT_JU} --debug
#else
#	echo "deploy file does not exist"
#fi

echo "end of the job"
