cd ./runs
for file in `ls $1`
do
	if expr index "$file" "Dec"
	then 
		echo $file
		rm -rf $file
	fi
done
cd ..
tensorboard --logdir runs --host=127.0.0.1 --port=6007
