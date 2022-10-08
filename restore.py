import pickle
import os
import sys
import wave

target = sys.argv[1]    #ファイルのpathがコマンドライン引数としてくる
AI_path = os.path.dirname(os.path.abspath(__file__)) + '/AI/restore_AI.sav'
restore_AI = pickle.load(AI_path)
with wave.open(target,'rb') as f:
    pass