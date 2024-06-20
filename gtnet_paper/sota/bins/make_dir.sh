fof=$1
dir=`basename $fof | cut -f3 -d.`
mkdir $dir
for f in `cat $fof`; do
    zcat $f > $dir/`basename $f .gz`
done
