# import user_data_pb2
from google.protobuf import json_format
from google.protobuf.json_format import ParseDict
from ht_utils import HTInputTokenizer

import pickle
# import user_preferences_pb2
import pos_dict_pb2
with open("postagger_model/subwordvocab.pkl","rb") as f:
    subwordvocab = pickle.load(f)
# 1. 원본 Python Dictionary
# pref_dict = {
#     "theme_mode": {"setting_name": "Theme Mode", "value": "dark"},
#     "notifications": {"setting_name": "Email Notifications", "value": "on"}
# }

# 2. Protobuf 메시지 객체 생성
posdict = pos_dict_pb2.PosDict()

# self.pos2index = {"[PAD]":0,"[SOS]":1,"[EOS]":2,"+":3,"O":4,"[UNK]":5}
# self.index2pos = {v:k for k,v in self.pos2index.items()}
        
# 3. Dictionary를 Map 필드에 직접 할당
for key, value in subwordvocab.pos2index.items():
    # print(key,value)
    # settings 맵 필드에 key로 접근하여 Value 메시지 객체를 생성하고 값을 할당합니다.
    # Protobuf는 이 과정에서 딕셔너리를 map의 value_type(PreferenceValue) 메시지로 자동 변환합니다.
    posdict.p2i[key] = value
    posdict.i2p[value] = key
    # tokdict.dict[key].value = value_dict["value"]

# for key,value in subwordvocab.index2pos.items():
#     posdict.
serialized_data = posdict.SerializeToString()

# 2. 파일에 쓰기 (바이너리 모드 'wb')
filename = "htpostagger.bin"
with open(filename, "wb") as f:
    f.write(serialized_data)