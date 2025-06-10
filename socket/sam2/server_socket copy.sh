
WORK_DIR='/home/lasker06/workspace/hojun/final_trans4mer/TranS4mer/trans4mer'
PYTHONPATH=${WORK_DIR} 

export PYTHONPATH=$PYTHONPATH
# data preprocessing
data_name=trans4mer
data_root=/home/lasker06/workspace/hojun/final_trans4mer/TranS4mer/${data_name}
file_pattern=/home/lasker06/workspace/hojun/final_trans4mer/TranS4mer/trans4mer/content/mytype.mp4


# data preprocessing
# data_name=video_poc
# data_root=/home/lasker01/jaehyun/datasets/${data_name}
# file_pattern="video/*"




# data_name=TranS4mer
# data_root=/home/lasker06/workspace/hojun/sbd/${data_name}
# file_pattern="content/*"



python3 ${WORK_DIR}/preprocessing/preprocessing_all.py \
    --root ${data_root} --pattern ${file_pattern}


ln -s ${data_root} ./data/${data_name}
# extract shot representation
# anno.ndjson 안에 있는 데이터에 대해서만 진행됨
LOAD_FROM=test
python3 ${WORK_DIR}/pretrain/extract_shot_repr.py \
		config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
		config.DATA_PATH="./data/${data_name}" \
	    +config.LOAD_FROM=${LOAD_FROM}


# inference
EXPR_NAME=test
LOAD_FROM=finetune/ckpt/finetune_${EXPR_NAME}/model-v1.ckpt
python3 ${WORK_DIR}/finetune/inference.py \
	config.TRAIN.BATCH_SIZE.effective_batch_size=1024 \
	config.TRAIN.NUM_WORKERS=8 \
	config.DISTRIBUTED.NUM_NODES=1 \
	config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
	config.EXPR_NAME=${EXPR_NAME} \
	config.DATA_PATH="./data/${data_name}" \
	+config.PRETRAINED_LOAD_FROM=${EXPR_NAME} \
	+config.FINETUNED_LOAD_FROM=${LOAD_FROM}

# time match, extract proper # of scene boundaries
outputroot=${WORK_DIR}/output/${EXPR_NAME}
inputjson=${WORK_DIR}/data/${data_name}/anno/anno.test.ndjson
topn=4
python3 ${WORK_DIR}/output/match_time_addrule.py \
    --outputroot ${outputroot} --inputjson ${inputjson} --topn ${topn}



# extract video
topn_ndjson=${outputroot}/top${topn}_result_time.ndjson
save_root=${data_root}/output_video
video_root=${data_root}/video #어차피 ndjson에 있는 video_id에 대해서만 함.
previous_s=15
last_s=15
python3 ${WORK_DIR}/output/extract_video.py \
    --topn_ndjson ${topn_ndjson} --save_root ${save_root} --video_root ${video_root} \
	--previous_s ${previous_s} --last_s ${last_s}
