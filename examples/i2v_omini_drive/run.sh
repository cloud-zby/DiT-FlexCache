python examples/i2v_omini_drive/infer_comotion_no_trans.py \
	--data_path "/nvfile-heatstorage/AIGC_H100/lbc/3.1_add_rq_lbc_selected_0916_updated.json" \
	--ref_image_path "/nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/ref_images/taiji.jpg" \
	--output_dir "/nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/inference_output.16fps.motion0918" \
	--infer_num 2 \
	--stride 1 \
	--size "720p" \
	--prompt "A cartoon-style character is dancing."

# A cartoon-style character is dancing.
# A realistic-style character is dancing.

# /nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/data/records_with_16fps_path_norepeat.pkl
# /nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/data/online_selected_0827_hz.pkl
# /nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/data/motion_16fps_pose_0912.pkl
# /nvfile-heatstorage/AIGC_H100/lbc/database/motion_3.1/250917/all_videos.json
