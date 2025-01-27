1. onnxruntime.dll

The Windows operating system possesses AI capabilities, utilizing onnxruntime as a component for inference in some of these capabilities.

`C:\Windows\System32\onnxruntime.dll`

However, there isn't a default program in the system to open onnxruntime model files. Therefore, [WinMLRunner.exe](https://github.com/Microsoft/Windows-Machine-Learning/releases) is used for model loading (and consequently, related crashes are not acknowledged by Microsoft). However, the Windows SDK may include APIs for directly handling model files.

You need to first install `pip install onnxruntime` to parse and mutate the model files.

Here's a mutation process for the file type:
Query relevant Python parsing libraries, use Python's reflection features to get the values of the parsed object's attributes, and then mutate them individually.

