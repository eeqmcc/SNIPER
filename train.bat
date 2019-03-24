rem Training a proposal network for 2 epochs
python main_train.py --cfg configs/faster/sniper_res101_e2e_cloth.yml --set TRAIN.USE_NEG_CHIPS False TRAIN.ONLY_PROPOSAL True TRAIN.end_epoch 2 output_path ./output/proposals
rem Extracting proposals on 2007_trainval for negative chip mining...
python main_test.py --cfg configs/faster/sniper_res101_e2e_cloth.yml --set TEST.EXTRACT_PROPOSALS True TEST.PROPOSAL_SAVE_PATH ./output/proposals TEST.TEST_EPOCH 2 output_path ./output/proposals dataset.test_image_set 2007_trainval
rem Training SNIPER with negative chip mining
python main_train.py --cfg configs/faster/sniper_res101_e2e_cloth.yml --set proposal_path ./output/proposals

