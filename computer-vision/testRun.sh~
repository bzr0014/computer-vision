COUNTER=0
MARDAS=""
while [ $COUNTER -le $# ]
do
	COUNTER=`expr $COUNTER + 1`
	MARDAS="$MARDAS $1"
	shift 
done
clear&clear&clear
rm -r Debug
mkdir Debug
cd Debug
cmake -G "Unix Makefiles" ../test
make
./code8-5 $MARDAS
