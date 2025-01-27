import onnxruntime
import numpy as np
import onnx
import sys
import subprocess
import sys
import shutil
# from google.protobuf.pyext._message import RepeatedScalarContainer
from google._upb._message import RepeatedScalarContainer
import copy

MODEL_FILE = "test.onnx"
if len(sys.argv) == 2:
    MODEL_FILE = sys.argv[1]

DZ = [0]
DP = [1, 2, 3, 7, 1024, 1337, 2**16-1, 2**16, 2**16+1, 2**20, 2**30, 2**31-1]
DBP = [2**31, 2**31+1, 2**32-1, 2**32, 2**32+1, 2**40, 2**63-1, 2**63, 2**63+1, 2**64-1, 2**64, 2**64+1]
DPP = [0, 1, 2, 3, 7, 1024, 1337, 2**16-1, 2**16, 2**16+1, 2**20, 2**30, 2**31-1, 2**31, 2**31+1, 2**32-1, 2**32, 2**32+1]
DN = [-i for i in DP]
DBN = [-i for i in DBP]
DI = DBN + DN + DZ + DP + DBP
DFN = [-1e32, -1., 0., 1., 1e20, 1e32]
DATA = DI


def run_model(fname, model):
    onnx.save(model, ".test.model")
    result = subprocess.run(["D:\\WinMLRunner.exe", "-model", ".test.model", "-GPU"], capture_output=True, text=True, timeout=30)
    if result.returncode in [2147500037, 0, 2147942487,2147500033]:
        pass
    else:
        shutil.move(".test.model", f"{result.returncode}_{fname}")
        print("====", result.returncode)

def mutate_model_manul(mfile):
    om = onnx.load(mfile)
    om.graph.node[0].attribute[3].ints[3] = -9223372036854775806       
    run_model("mu", om)

def mutate_model(mfile):
    om = onnx.load(mfile)
    mutate_cnt = 1
    for dim_value in DI[::-1]:
        ops = ["Conv"]
        # ops = ["Conv", "MaxPool",'Transpose', 'Split', 'ReduceMean', 'Unsqueeze', 'ReduceSum', 'Squeeze','QLinearConv','Slice', 'QLinearAveragePool']
        for ni,node in enumerate(om.graph.node):
            if node.op_type in ops:
                continue
            else:
                ops.append(node.op_type)
            if hasattr(node, "attribute"):
                for ai,attr in enumerate(node.attribute):
                    if hasattr(attr, "ints"):
                        omnew = copy.deepcopy(om)
                        if isinstance(attr.ints, RepeatedScalarContainer):
                        # print(type(attr.ints))
                        # if isinstance(attr.ints, list):
                            for i in range(len(attr.ints)):
                                try:
                                    omnew.graph.node[ni].attribute[ai].ints[i] = dim_value
                                    print(node.name, node.op_type)
                                    fname = f"omnew.graph.node_{ni}.attribute_{ai}.ints_{i}={dim_value}"
                                    run_model(fname, omnew)
                                except Exception as e:
                                    print(e)
                                    continue
                        else:
                            attr.ints = dim_value
                            fname = f"omnew.graph.node_{ni}.attribute_{ai}.ints={dim_value}"
                            run_model(fname, om)

mutate_model(MODEL_FILE)

'''
/features/features.2/MaxPool
omnew.graph.node[2].attribute[3].ints[0] = -2147483648
Floating point exception


/features/features.0/Conv
omnew.graph.node[0].attribute[3].ints[3] = -9223372036854775807
Segmentation fault


/avgpool/AveragePool
omnew.graph.node[13].attribute[1].ints[0] = -3
Floating point exception
'''