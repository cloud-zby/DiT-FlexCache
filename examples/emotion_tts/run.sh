# python main_multiprocessing.py --task zero_shot --ref-speech-path "/gemini/space/jrh/programme/ScoreTTS/2月14日 (2)(5).MP3" --texts-to-synthesize "这些年你背着天尊做了什么，当我不知道吗。" --output-dir tts_results_zeroshot

# python main_multiprocessing.py --task emotion --need-emotion SAD  --ref-speech-path "/gemini/space/jrh/programme/ScoreTTS/2月14日 (2)(5).MP3" --texts-to-synthesize "这些年你背着天尊做了什么，当我不知道吗。" --output-dir tts_results_SAD

# python main_multiprocessing.py --task emotion --need-emotion HAPPY  --ref-speech-path "/gemini/space/jrh/programme/ScoreTTS/2月14日 (2)(5).MP3" --texts-to-synthesize "这些年你背着天尊做了什么，当我不知道吗。" --output-dir tts_results_HAPPY

# python main_multiprocessing.py --task emotion --need-emotion ANGRY  --ref-speech-path "/gemini/space/jrh/programme/ScoreTTS/2月14日 (2)(5).MP3" --texts-to-synthesize "这些年你背着天尊做了什么，当我不知道吗。" --output-dir tts_results_ANGRY
CURRENT_DIR=$(pwd)
echo "Current directory: $CURRENT_DIR"
cd $CURRENT_DIR/teleai_pipe/pipelines/tele_emotts   # 切换路径

python main_multiprocessing.py \
    --task emotion \
    --need-emotion SURPRISED \
    --ref-speech-path "/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/2月14日 (2)(5).MP3"\
    --texts-to-synthesize "这些年你背着天尊做了什么，当我不知道吗。" \
    --output-dir ${CURRENT_DIR}/examples/emotion_tts/tts_results_SURPRISED

# python main_multiprocessing.py \
#     --task zero_shot \
#     --ref-speech-path "/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/2月14日 (2)(5).MP3"\
#     --texts-to-synthesize "这些年你背着天尊做了什么，当我不知道吗。" \
#     --output-dir ${CURRENT_DIR}/examples/emotion_tts/tts_results_zero_shot