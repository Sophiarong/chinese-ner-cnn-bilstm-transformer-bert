python run_cnn.py \
  --device=0\
  --batch_size=128 \
  --epoch=300 \
  --lr=0.001 \
  --train_file=/home/rong/work/NER/homework/shouxie/data/msr_training_split.txt \
  --dev_file=/home/rong/work/NER/homework/shouxie/data/msr_dev_split.txt \
  --output_file=/home/rong/work/NER/homework/shouxie/data/msr_output.txt \
  --test_file=/home/rong/work/NER/homework/shouxie/data/msr_test.txt