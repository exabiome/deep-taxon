conda activate cat 

FA=$1
echo $FA

BASE=`basename $FA _genomic.fna.gz`
DIR=`dirname $FA`

LOG="$BASE.prodigal_infer.log"
OUT="${BASE}_transorfs.faa"
CMD="prodigal -a $OUT"

echo $CMD > $LOG
date +BEFORE_%Y-%m-%dT%H:%M:%S_%s >> $LOG 2>&1
zcat $FA | $CMD > /dev/null 2>> $LOG
date +AFTER_%Y-%m-%dT%H:%M:%S_%s >> $LOG 2>&1
gzip $OUT
mv $OUT.gz $LOG $DIR
