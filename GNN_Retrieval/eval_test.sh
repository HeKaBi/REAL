SPLIT="test"
DATASET_LIST="RoG-cwq"
MODEL_NAME=chatgpt
PROMPT_PATH=llm/prompts/llama2_predict.txt
BEAM_LIST="3" # "1 2 3 4 5"d
OUTPUT_BASE_PATH='./myresults'
export OPENAI_API_KEY='EMPTY'
export OPENAI_API_BASE='http://0.0.0.0:8000/v1'

#GNN-RAG
for DATA_NAME in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        RULE_PATH=results/gen_rule_path/${DATA_NAME}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl
        RULE_PATH_G1=checkpoint/pretrain/prn_cwq-rearev-lmsr_test.info
        RULE_PATH_G2=None #results/gnn/${DATA_NAME}/rearev-lmsr/test.info

        # no rog
        python llm/src/qa_prediction/predict_answer.py \
            --model_name ${MODEL_NAME} \
            -d ${DATA_NAME} \
            --prompt_path ${PROMPT_PATH} \
            --rule_path ${RULE_PATH} \
            --rule_path_g1 ${RULE_PATH_G1} \
            --rule_path_g2 ${RULE_PATH_G2} \
            --predict_path ${OUTPUT_BASE_PATH}\
            --force
            
    done
done
