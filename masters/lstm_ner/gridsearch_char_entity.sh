#!/bin/bash
#SBATCH --partition gpgpuC --gres=gpu:1
export PATH=/vol/bitbucket/starter/bin/:$PATH
#export PATH=/vol/bitbucket/vnp23/gaml_venv/bin/:$PATH
export PATH=/vol/bitbucket/vnp23/miniconda3/bin:$PATH
export ANNLOC=/vol/bitbucket/astronlp/consensus
export MODELLOC=/vol/bitbucket/vnp23/model_attempts/bert_lstm_ner/tuning/models/
export BERTMODEL=/vol/bitbucket/vnp23/model_attempts/train_bert_embeds/model/astrobert_m256_n2500/astroBERT_2500.pt
export REPODIR=/homes/vnp23/Masters/gaml-reproduction
export SCRIPTPATH=vishnu/lstm_ner/lstmcharentitymodel.py
export SCRIPTLOC=${REPODIR}/${SCRIPTPATH}
source /vol/bitbucket/vnp23/miniconda3/etc/profile.d/conda.sh
conda activate py38
source /vol/cuda/10.0.130/setup.sh
export PYTHONPATH=$PYTHONPATH:${REPODIR}/
TERM=vt100 # or TERM=xterm

export BERTMODELBASE=adsabs/astroBERT


echo -e "\n\n\n\n\n\n\n\n"
echo -e "\n******************************************************************************"
echo -e "\n**********************************STARTING************************************"
echo -e "\n******************************************************************************"
python3 -V

echo -e "\n\n\n"

echo "Start scheduling entity models."

## Character Encoding LSTM Entity Model
for hidden in 64 128 512 1024
do
	for layers in 1 2 3
	do
		for charemb in 32 64 128
		do
			for weight in "--class-weight" ""
			do
				MODELNAME=LSTMCharsEntity_"$hidden"_"$layers"_"$charemb""$([ -n "$weight" ] && echo "_W")"
				echo python3 $SCRIPTLOC $ANNLOC $BERTMODEL $MODELLOC/$MODELNAME -M $BERTMODELBASE $weight --hidden "$hidden" --layers "$layers" --char-emb "$charemb"
				python3 $SCRIPTLOC $ANNLOC $BERTMODEL $MODELLOC/$MODELNAME -M $BERTMODELBASE $weight --hidden "$hidden" --layers "$layers" --char-emb "$charemb"
			done
		done
	done
done


echo "Finish scheduling entity models."




echo -e "\n\n\n\n\n\n\n\n"
echo -e "\n******************************************************************************"
echo -e "\n**********************************COMPLETE************************************"
echo -e "\n******************************************************************************"
uptime