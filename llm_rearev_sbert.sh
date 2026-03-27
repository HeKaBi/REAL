  export OPENAI_API_KEY=EMPTY
  export OPENAI_API_BASE=http://127.0.0.1:8000/v1

  python llm/src/qa_prediction/predict_answer.py \
    --model_name chatgpt \
    -d RoG-webqsp \
    --prompt_path llm/prompts/llama2_predict.txt \
    --rule_path_g1 llm/results/gnn/RoG-webqsp/rearev-sbert/test.info \
    --rule_path_g2 None \
    --predict_path myresults \
    --force