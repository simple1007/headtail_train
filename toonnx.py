import os
import torch
import pickle

from postagger_model.model2 import PosLSTMMini2CPP
from tokenizer_model.model2 import TK_Model_Mini2CPP
root = __file__

root = root.split(os.sep)
root = os.sep.join(root[:-1])
# print(root)
os.environ['HT'] = root + os.sep + "postagger_model"
def to_onnx(model,filename,unfold_=False):
    x = torch.tensor(torch.randint(0,100,(1,260),requires_grad=True,dtype=torch.float).detach().numpy().tolist())#.to(device)#.to(device)
    unfold = x
    
    print(x.shape)
    print(unfold.shape)
    if unfold_:
        torch_out = model(x,unfold)
    else:
        torch_out = model(x)
    
    # options = torch.onnx.ExportOptions(dynamic_shapes=True)
    # ex = torch.onnx.dynamo_export(model, (x), **input_dict, export_options=options)

    # print("Running torch.export.export ...")
    # x1 = torch.export.Dim("x1")
    # x2 = torch.export.Dim("x2")
    # # x3 = torch.export.Dim("x3")
    # # b1 = torch.export.Dim("b1")
    # # b2 = torch.export.Dim("b2")
    # # b3 = torch.export.Dim("b3")
    # dynamic_shapes={"input":{0:x1,1:x2}}#{'input_ids': {0: b1, 1: x1}, 'attention_mask': {0: b2, 1: x2}, 'token_type_ids': {0: b3, 1: x3}}
    # ex_model = torch.export.export(
    #     model,
    #     tuple([x]),
    #     # kwargs={"input":x},
    #     # dynamic_shapes=dynamic_shapes,
    #     # strict=False
    # )
    # return 
    print(torch_out[0].shape)
    # exit()
    # dynamic_shapes={'input_ids': {0: b1, 1: x1}, 'attention_mask': {0: b2, 1: x2}, 'token_type_ids': {0: b3, 1: x3}}

    # ex_model = torch.export.export(
    #     model,
    #     tuple(input_list),
    #     kwargs=input_dict,
    #     dynamic_shapes=dynamic_shape,
    #     strict=False
    # )
    """from torch.export import Dim
    onnx_program = torch.export.dynamo_export(
        model,
        (x),
        dynamic_shapes={
            0: {0: Dim.DYNAMIC,1:Dim.DYNAMIC}
        },
        input_names = ['input'],#"unfold"],   # 모델의 입력값을 가리키는 이름
        output_names = ['output'], # 모델의 출력값을 가리키는 이름
    )
    onnx_program.save(filename)
    return"""
    if unfold_:
        inputs = (x,unfold)
        inputs_name = ["input","unfold"]
        dynamic_shape = {
            'input' : {0 : 'batch_size',1:"sentence_length"},    # 가변적인 길이를 가진 차원
            'unfold' : {0 : 'batch_size',1:"sentence_length"},
            'output' : {0 : 'batch_size',1:"sentence_length"}
        }
    else:
        inputs = (x)
        inputs_name = ["input"]
        dynamic_shape = {
            'input' : {0 : 'batch_size',1:"sentence_length"},    # 가변적인 길이를 가진 차원
            # 'unfold' : {0 : 'batch_size',1:"sentence_length"},
            'output' : {0 : 'batch_size',1:"sentence_length"}
        }
    torch.onnx.export(
        model,               # 실행될 모델
        inputs,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
        filename,   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
        export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
        opset_version=17,          # 모델을 변환할 때 사용할 ONNX 버전
        do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
        input_names = inputs_name,#['input'],#"unfold"],   # 모델의 입력값을 가리키는 이름
        output_names = ['output'], # 모델의 출력값을 가리키는 이름
        dynamic_axes=dynamic_shape,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL 
    )

def to_onnx_wautospace_tok(model,filename):
    x = torch.tensor(torch.randint(0,100,(1,260),requires_grad=True,dtype=torch.float).detach().numpy().tolist())#.to(device)#.to(device)
    unfold = x
    
    print(x.shape)
    print(unfold.shape)
    torch_out = model(x,unfold)
    print(torch_out[0].shape)
    # exit()
    torch.onnx.export(
        model,               # 실행될 모델
        (x,unfold),                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
        filename,   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
        export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
        opset_version=23,          # 모델을 변환할 때 사용할 ONNX 버전
        do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
        input_names = ['input',"unfold"],   # 모델의 입력값을 가리키는 이름
        output_names = ['output',"output2"], # 모델의 출력값을 가리키는 이름
        dynamic_axes={
            'input' : {0 : 'batch_size',1:"sentence_length"},    # 가변적인 길이를 가진 차원
            'unfold' : {0 : 'batch_size',1:"sentence_length"},
            'output' : {0 : 'batch_size',1:"sentence_length"},
            'output2' : {0 : 'batch_size',1:"sentence_length"}
        }
    )
###postagger
if False:
    print("-----model loading")
    with open(os.path.join(os.environ["HT"],"transformer_encoder3_small","subwordvocab.pkl"),"rb") as f:
        subwordvocab = pickle.load(f)
    print("pad",subwordvocab.__dict__["pad"])
    
    print("cls",subwordvocab.__dict__["cls"])
    
    print("cls",subwordvocab.__dict__["sep"])
    
    print("space",subwordvocab.__dict__["space"])
    print("outpad",subwordvocab.__dict__["outpad"])
    print("pos2index_len",len(subwordvocab.__dict__["pos2index"]))
    # print(subwordvocab.__dict__["tags"])
    # exit()
    
    model = PosLSTMMini2CPP(pad=subwordvocab.__dict__["pad"],outsize=len(subwordvocab.__dict__["pos2index"]))
    model.load_state_dict(torch.load(os.path.join(os.environ["HT"],"transformer_encoder3_small","model_20"))["model"])
    print("-----model to onnx")
    to_onnx(model,"ht_postagger.onnx")
    print("-----model onnx to qint4")
    # os.system("C:\\Users\\ty341\\Desktop\\headtail_train\\.venv\\Scripts\\activate.bat")
    print("python -m onnxruntime.quantization.matmul_bnb4_quantizer --input_model ht_postagger.onnx --output_model pos-model-uint8.onnx")
    ###tokenizer

if True:
    print("-----model loading")
    with open('tokenizer_model/lstm_vocab.pkl','rb') as f:
        lstm_vocab = pickle.load(f)
    max_len = 400
    model = TK_Model_Mini2CPP(max_len,lstm_vocab,None)
    print(torch.load(os.path.join("tokenizer_model","transformer_pre_encoder_small","model_14")).keys())
    # exit()    
    model.load_state_dict(torch.load(os.path.join("tokenizer_model","transformer_pre_encoder_small","model_14"))["model"])
    print("-----model to onnx")
    to_onnx(model,"ht_tokenizer.onnx")
    
    print("-----model onnx to qint4")
    # os.system("C:\\Users\\ty341\\Desktop\\headtail_train\\.venv\\Scripts\\activate.bat")
    print("python -m onnxruntime.quantization.matmul_bnb4_quantizer --input_model ht_tokenizer.onnx --output_model tok-model-uint8.onnx")
    # print("python -m onnxruntime.quantization.static_quantize_runner -i ht_tokenizer.onnx -o ht_tokenizer_qint4.onnx --activation_type qint4 --weight_type qint4")
# exit()
import onnx
import onnxruntime as ort
# model = onnx.load("ht_tokenizer.onnx")
EP_list = ['CPUExecutionProvider']
sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 2
# sess_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess = ort.InferenceSession("ht_postagger.onnx",sess_options=sess_opt, providers=EP_list)
# ort_session = ort.InferenceSession(
#     "ht_tokenizer.onnx", providers=["CPUExecutionProvider"]
# )

# # onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}
import numpy as np
# # ONNX Runtime returns a list of outputs
# x = torch.tensor(torch.randint(0,100,(1,300),requires_grad=True,dtype=torch.float).detach().numpy().tolist()).numpy()#.to(device)#.to(device)
a = [6. ,56. ,19. ,3. ,20. ,1304. ,19. ,3., 21. ,257. ,19., 3., 28. ,509. ,19. ,3. ,27. ,1202. ,19. ,3. ,23., 7.]
# a = [a]
x = np.array([a],dtype=np.float32)
res = sess.run(None,{"input":x})
print(res)
# print(res)
# # x = torch.tensor(torch.randint(0,100,(1,260),requires_grad=True,dtype=torch.float).detach().numpy().tolist())#.to(device)#.to(device)
    