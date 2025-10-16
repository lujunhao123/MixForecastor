export CUDA_VISIBLE_DEVICES=0
for model_name in PatchTST
do
for pl in 3
do

python -u run_ramp.py \
--Ramp_name MixRamp \
--site NSW1_30min \
--train_epochs 200 \
--optype Pcgrad \
--is_training 1 \
--model_id NSW1_30min_48_${pl} \
--model $model_name \
--data NSW1_30min \
--features M \
--seq_len 48 \
--label_len 12 \
--pred_len ${pl} \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--enc_in 9 \
--dec_in 9 \
--c_out 9 \
--des 'Exp' \
--itr 1

python -u run_ramp.py \
--Ramp_name SNoramlRamp \
--Singel_task Con \
--site NSW1_30min \
--train_epochs 200 \
--is_training 1 \
--model_id NSW1_30min_48_${pl} \
--model $model_name \
--data NSW1_30min \
--features M \
--seq_len 48 \
--label_len 12 \
--pred_len ${pl} \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--enc_in 9 \
--dec_in 9 \
--c_out 9 \
--des 'Exp' \
--itr 1


python -u run_ramp.py \
--Ramp_name SNoramlRamp \
--Singel_task Dis \
--site NSW1_30min \
--train_epochs 200 \
--is_training 1 \
--model_id NSW1_30min_48_${pl} \
--model $model_name \
--data NSW1_30min \
--features M \
--seq_len 48 \
--label_len 12 \
--pred_len ${pl} \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--enc_in 9 \
--dec_in 9 \
--c_out 9 \
--des 'Exp' \
--itr 1

python -u run_ramp.py \
--Ramp_name MNormalRamp \
--site NSW1_30min \
--train_epochs 200 \
--is_training 1 \
--model_id NSW1_30min_48_${pl} \
--model $model_name \
--data NSW1_30min \
--features M \
--seq_len 48 \
--label_len 12 \
--pred_len ${pl} \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--enc_in 9 \
--dec_in 9 \
--c_out 9 \
--des 'Exp' \
--itr 1

done
done