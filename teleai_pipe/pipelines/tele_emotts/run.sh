# python main_multiprocessing.py --task zero_shot --ref-speech-path "/gemini/space/jrh/programme/ScoreTTS/2月14日 (2)(5).MP3" --texts-to-synthesize "这些年你背着天尊做了什么，当我不知道吗。" --output-dir tts_results_zeroshot

python main_multiprocessing.py \
    --task emotion \
    --need-emotion SAD \
    --ref-speech-path "/gemini/space/jrh/programme/ScoreTTS/2月14日 (2)(5).MP3" \
    --texts-to-synthesize "这些年你背着天尊做了什么，当我不知道吗。" \
    --output-dir tts_results_SAD

