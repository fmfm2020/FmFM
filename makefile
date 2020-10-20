data_process:
	cd data/criteo/
	wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz
	tar -zxvf dac.tar.gz
	python trains_criteo_dataset.py
	shuf all_data.csv all_data.tmp
	mv all_data.tmp all_data.csv
	head -n 36672493 all_data.csv > train.csv
	tail -n 1327180 all_data.csv > test.csv
	tail -n 2654360 all_data.csv | head -n 1327180 > val.csv
run:
	sh start_train_criteo.sh