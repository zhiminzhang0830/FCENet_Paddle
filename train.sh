# recommended paddle.__version__ == 2.0.0
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml

python -m paddle.distributed.launch --gpus '1'  tools/train.py -c configs/det/det_r50_fce_ctw_adam.yml

# python tools/infer_det.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0_jw_multi_infer.yml
