#!/bin/bash
#SBATCH --partition gpgpuC --gres=gpu:1
export PATH=/vol/bitbucket/vnp23/miniconda3/bin:$PATH
source /vol/bitbucket/vnp23/miniconda3/etc/profile.d/conda.sh
conda activate py38
export ANNLOC=/vol/bitbucket/astronlp/consensus
export ANNCONFIGLOC=/homes/vnp23/Masters/gaml-reproduction/annotations/annotation.conf
export W2VLOC=/vol/bitbucket/astronlp/arXivWord2Vec.txt
export MODELLOC=/vol/bitbucket/vnp23/model_attempts/w2v_lstm_rext/sep_models/justbinary/
export REPODIR=/homes/vnp23/Masters/gaml-reproduction/
export TFIDFLOC=/vol/bitbucket/vnp23/data/tfidf_none_100000.pkl
export SCRIPTLOC=vishnu/rel_extract/lstm/lstm_rel_binary.py
export SCRIPTPATH=${REPODIR}${SCRIPTLOC}
source /vol/cuda/10.0.130/setup.sh
export PYTHONPATH=$PYTHONPATH:$REPODIR
TERM=vt100 # or TERM=xterm


echo -e "\n\n\n\n\n\n\n\n"
echo -e "\n******************************************************************************"
echo -e "\n**********************************STARTING************************************"
echo -e "\n******************************************************************************"
python3 -V

echo -e "\n\n\n"

echo "Start scheduling relation models."

## LSTM Relation Model
for pad in 3 5 9 15
do
	for hidden in "512" "128" "64" "128-64" "512-128-64"
	do
		for weight in "--class-weight" ""
		do
		  for relation in "Name" "Measurement" "Confidence" "Property"
		  do
        MODELNAME=LSTMRelation_"$pad"_"$hidden""$([ -n "$weight" ] && echo "_W")"
        echo python3 $SCRIPTPATH $ANNLOC $W2VLOC $MODELLOC -b 6000 $weight --window-pad "$pad" --hidden "$hidden" --relations "$relation"
        python3 $SCRIPTPATH $ANNLOC $W2VLOC $MODELLOC/$MODELNAME -b 6000 $weight --window-pad "$pad" --hidden "$hidden" --relations "$relation"
      done
		done
	done
done

echo "Finish scheduling relation models."




echo -e "\n\n\n\n\n\n\n\n"
echo -e "\n******************************************************************************"
echo -e "\n**********************************COMPLETE************************************"
echo -e "\n******************************************************************************"
uptime