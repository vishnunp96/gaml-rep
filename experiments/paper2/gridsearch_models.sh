## Wrapped in braces so we can redirect output from within file
RUNDESIGNATION="$(date +"%Y-%m-%d-%H%M%S")"
CLUSTERDIR=/cluster/project2/gaml/trained/paper2/"$RUNDESIGNATION"_run/
LOGFILE="$CLUSTERDIR"/"$RUNDESIGNATION"_scheduling.log
mkdir -p "$CLUSTERDIR"
LOGDIR="$CLUSTERDIR"/"$RUNDESIGNATION"_logs/
mkdir -p "$LOGDIR"
{
MODELDIR="$HOME/Documents/gaml/annotations/models/"
ANNOTATIONTYPES="MeasuredValue,Constraint,ParameterSymbol,ParameterName,ConfidenceLimit,ObjectName,Confidence,Measurement,Name,Property,UpperBound,LowerBound"

ANNDIR="/mnt/annotations/annotationProject/backup2/consensus/"

SEED=42
DATASPLIT="60-20-20"
EMBEDDINGS="/mnt/trained/arXivNeatWord2Vec.txt"
STANDOFFCONFIG="/mnt/annotations/annotation.conf"

OUTPUTDIR="/mnt/trained/paper2/""$RUNDESIGNATION""_run/"
ENTITYDIR="$OUTPUTDIR/entityModels/"
RELATIONDIR="$OUTPUTDIR/relationModels/"
ATTRIBUTEDIR="$OUTPUTDIR/attributeModels/"

## Check output model directories exist
#mkdir -p "$ENTITYDIR"
#mkdir -p "$RELATIONDIR"
#mkdir -p "$ATTRIBUTEDIR"
#echo "Output model directories created."

##### Entity Models ######

echo "Start scheduling entity models."

## Deep Entity Model
for pad in 3 5 9 15
do
	for hidden in "1024" "512" "128" "512-128" "1024-512-128"
	do
		for weight in "--class-weight" ""
		do
			MODELNAME=DeepEntity_"$pad"_"$hidden""$([ -n "$weight" ] && echo "_W")"
			echo qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/deepindexwindowmemoryentitymodel.py $ANNDIR $EMBEDDINGS "$ENTITYDIR/$MODELNAME" $weight --window-pad "$pad" --hidden "$hidden" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
			qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/deepindexwindowmemoryentitymodel.py $ANNDIR $EMBEDDINGS "$ENTITYDIR/$MODELNAME" $weight --window-pad "$pad" --hidden "$hidden" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
		done
	done
done

## Smoothed Deep Entity Model
for pad in 3 5 9 15
do
	for hidden in "1024" "512" "128" "512-128" "1024-512-128"
	do
		for weight in "--class-weight" ""
		do
			MODELNAME=SmoothedEntity_"$pad"_"$hidden""$([ -n "$weight" ] && echo "_W")"
			echo qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/indexsmoothedwindowmemoryentitymodel.py $ANNDIR $EMBEDDINGS "$ENTITYDIR/$MODELNAME" $weight --window-pad "$pad" --hidden "$hidden" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
			qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/indexsmoothedwindowmemoryentitymodel.py $ANNDIR $EMBEDDINGS "$ENTITYDIR/$MODELNAME" $weight --window-pad "$pad" --hidden "$hidden" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
		done
	done
done

## Character Encoding Deep Entity Model
for pad in 3 5 9 15
do
	for hidden in "1024" "512" "128" "512-128" "1024-512-128"
	do
		for charemb in 32 64 128
		do
			for weight in "--class-weight" ""
			do
				MODELNAME=DeepCharsEntity_"$pad"_"$hidden"_"$charemb""$([ -n "$weight" ] && echo "_W")"
				echo qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/testcharentitymodel.py $ANNDIR $EMBEDDINGS "$ENTITYDIR/$MODELNAME" $weight --window-pad "$pad" --hidden "$hidden" --char-emb "$charemb" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
				qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/testcharentitymodel.py $ANNDIR $EMBEDDINGS "$ENTITYDIR/$MODELNAME" $weight --window-pad "$pad" --hidden "$hidden" --char-emb "$charemb" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
			done
		done
	done
done

## LSTM Entity Model
for hidden in 64 128 512 1024
do
	for layers in 1 2 3
	do
		for weight in "--class-weight" ""
		do
			MODELNAME=LSTMEntity_"$hidden"_"$layers""$([ -n "$weight" ] && echo "_W")"
			echo qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/lstmentitymodel.py $ANNDIR $EMBEDDINGS "$ENTITYDIR/$MODELNAME" $weight --hidden "$hidden" --layers "$layers" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
			qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/lstmentitymodel.py $ANNDIR $EMBEDDINGS "$ENTITYDIR/$MODELNAME" $weight --hidden "$hidden" --layers "$layers" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
		done
	done
done

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
				echo qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/lstmcharentitymodel.py $ANNDIR $EMBEDDINGS "$ENTITYDIR/$MODELNAME" $weight --hidden "$hidden" --layers "$layers" --char-emb "$charemb" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
				qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/lstmcharentitymodel.py $ANNDIR $EMBEDDINGS "$ENTITYDIR/$MODELNAME" $weight --hidden "$hidden" --layers "$layers" --char-emb "$charemb" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
			done
		done
	done
done


echo "Finish scheduling entity models."


##### Relation Models ######

echo "Start scheduling relation models."

## LSTM Relation Model
for pad in 3 5 9 15
do
	for hidden in "512" "128" "64" "128-64" "512-128-64"
	do
		for weight in "--class-weight" ""
		do
			MODELNAME=LSTMRelation_"$pad"_"$hidden""$([ -n "$weight" ] && echo "_W")"
			echo qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/indexsplitspansdirectedrelationmodel.py $ANNDIR $EMBEDDINGS "$RELATIONDIR/$MODELNAME" "$STANDOFFCONFIG" $weight --window-pad "$pad" --hidden "$hidden" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
			qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/indexsplitspansdirectedrelationmodel.py $ANNDIR $EMBEDDINGS "$RELATIONDIR/$MODELNAME" "$STANDOFFCONFIG" $weight --window-pad "$pad" --hidden "$hidden" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
		done
	done
done

## Rules Based Model
MODELNAME=RulesRelation
echo qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/rulesrelationmodel.py "$ANNDIR" "$RELATIONDIR/$MODELNAME" "$STANDOFFCONFIG" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/rulesrelationmodel.py "$ANNDIR" "$RELATIONDIR/$MODELNAME" "$STANDOFFCONFIG" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"

echo "Finish scheduling relation models."


##### Attribute Models ######

echo "Start scheduling attribute models."

## Deep Attribute Model
for pad in 3 5 9 15
do
	for hidden in "1024" "512" "128" "512-128" "1024-512-128"
	do
		for weight in "--class-weight" ""
		do
			MODELNAME=DeepAttribute_"$pad"_"$hidden""$([ -n "$weight" ] && echo "_W")"
			echo qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/windowattributemodel.py $ANNDIR $EMBEDDINGS "$ATTRIBUTEDIR/$MODELNAME" $weight --window-pad "$pad" --hidden "$hidden" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
			qsub -N $MODELNAME -o "$LOGDIR/$MODELNAME.o" -e "$LOGDIR/$MODELNAME.e" singlerun.job $MODELDIR/windowattributemodel.py $ANNDIR $EMBEDDINGS "$ATTRIBUTEDIR/$MODELNAME" $weight --window-pad "$pad" --hidden "$hidden" --seed "$SEED" --split "$DATASPLIT" --types "$ANNOTATIONTYPES"
		done
	done
done

echo "Finish scheduling attribute models."

echo "Ending scheduling."
} > $LOGFILE
