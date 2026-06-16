# generation + refinement (one split, one refiner)
python run_x_to_c_to_y.py --dataset PH2 --split 0 --model Explicd \
    --concept_extractor Explicd --generate_concepts --data_path data \
    --refiner mistral --margin_threshold 0.2

# classifier path — this is what would crash if MMed has no chat template
python run_x_to_c_to_y.py --dataset PH2 --split 0 --concept_extractor Explicd \
    --llm MMed --ckpt $MMED_CKPT --n_demos 0 --refiner mistral
