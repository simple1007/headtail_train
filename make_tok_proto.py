# import user_data_pb2
from google.protobuf import json_format
from google.protobuf.json_format import ParseDict


import pickle
# import user_preferences_pb2
import tok_dict_pb2
with open("tokenizer_model/lstm_vocab.pkl","rb") as f:
    lstm_vocab = pickle.load(f)
# 1. 원본 Python Dictionary
# pref_dict = {
#     "theme_mode": {"setting_name": "Theme Mode", "value": "dark"},
#     "notifications": {"setting_name": "Email Notifications", "value": "on"}
# }

# 2. Protobuf 메시지 객체 생성
tokdict = tok_dict_pb2.TokDict()

# 3. Dictionary를 Map 필드에 직접 할당
for key, value in lstm_vocab.items():
    # print(key,value)
    # settings 맵 필드에 key로 접근하여 Value 메시지 객체를 생성하고 값을 할당합니다.
    # Protobuf는 이 과정에서 딕셔너리를 map의 value_type(PreferenceValue) 메시지로 자동 변환합니다.
    tokdict.dict[key] = value
    # tokdict.dict[key].value = value_dict["value"]


serialized_data = tokdict.SerializeToString()

# 2. 파일에 쓰기 (바이너리 모드 'wb')
filename = "httokenizer.bin"
with open(filename, "wb") as f:
    f.write(serialized_data)