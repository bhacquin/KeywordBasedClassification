
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

DATASET=agnews
LABEL_NAME_FILE=label_name_loop.txt
TRAIN_CORPUS=train.txt
TEST_CORPUS=test.txt
TEST_LABEL=test_labels.txt
TRAIN_LABEL=train_labels.txt
LOOP_OVER_VOCAB=3
CATE_VOCAB_SIZE=100
TOP_PRED_NUM=50
MATCH_THRESH=20
MAX_LEN=200
TRAIN_BATCH=32
ACCUM_STEP=4
EVAL_BATCH=128
GPUS=1
MCP_EPOCH=3
SELF_TRAIN_EPOCH=1

for keyword in politics sports business technology
do
	for number_of_loop in 1 2 3
	do 
		echo $keyword$number_of_loop
		mkdir -p datasets/agnews/$keyword$number_of_loop
		echo $keyword > datasets/agnews/${LABEL_NAME_FILE}
		LOOP_OVER_VOCAB=$number_of_loop
		echo "Building vocab"
		python3 src/train.py --dataset_dir datasets/${DATASET}/ --label_names_file ${LABEL_NAME_FILE} \
                    --train_file ${TRAIN_CORPUS} --train_label_file ${TRAIN_LABEL}\
                    --category_vocab_size ${CATE_VOCAB_SIZE}\
		    --match_threshold ${MATCH_THRESH}\
		    --top_pred_num ${TOP_PRED_NUM}\
		    --loop_over_vocab ${LOOP_OVER_VOCAB}\
		    --test_file ${TEST_CORPUS} --test_label_file ${TEST_LABEL} \
                    --max_len ${MAX_LEN} \
                    --train_batch_size ${TRAIN_BATCH} --accum_steps ${ACCUM_STEP} --eval_batch_size ${EVAL_BATCH} \
                    --gpus ${GPUS} \
                    --mcp_epochs ${MCP_EPOCH} --self_train_epochs ${SELF_TRAIN_EPOCH} 
		mv datasets/agnews/*.pt datasets/agnews/$keyword$number_of_loop
		echo "Running Model"
		python3 datasets/agnews/run.py --keyword $keyword --number_of_loop $number_of_loop
	done
done
