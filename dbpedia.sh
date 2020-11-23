
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

DATASET=dbpedia
LABEL_NAME_FILE=label_names_1.txt
TRAIN_CORPUS=train.txt
TEST_CORPUS=test.txt
TEST_LABEL=test_labels.txt
TRAIN_LABEL=train_labels.txt
MAX_LEN=200
TRAIN_BATCH=16
ACCUM_STEP=8
EVAL_BATCH=128
GPUS=1
MCP_EPOCH=3
SELF_TRAIN_EPOCH=1

python src/train.py --dataset_dir datasets/${DATASET}/ --label_names_file ${LABEL_NAME_FILE} \
                    --train_file ${TRAIN_CORPUS} --train_label_file ${TRAIN_LABEL}\
                    --test_file ${TEST_CORPUS} --test_label_file ${TEST_LABEL} \
                    --max_len ${MAX_LEN} \
                    --train_batch_size ${TRAIN_BATCH} --accum_steps ${ACCUM_STEP} --eval_batch_size ${EVAL_BATCH} \
                    --gpus ${GPUS} \
                    --mcp_epochs ${MCP_EPOCH} --self_train_epochs ${SELF_TRAIN_EPOCH} \
