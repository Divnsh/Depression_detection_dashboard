if ps -ef | grep -v grep | grep tensorflow_model_server ; then 
	exit 0 
else 
	nohup tensorflow_model_server --rest_api_port=8501 --model_name=roberta_dep --model_base_path='gs://saved_models19082020/my_models' & > /dev/null
	exit 0
fi 
