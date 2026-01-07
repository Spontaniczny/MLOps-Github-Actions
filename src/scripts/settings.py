from dataclasses import dataclass

@dataclass
class Settings:
    s3_bucket_name = "mlops-lab11-models-kaspersky"
    s3_model_dir = "model/"
    local_model_dir = "model"
    classifier_joblib_path = local_model_dir + "/classifier.joblib"
    onnx_classifier_path = "model_onnx/classifier.onnx"
    onnx_embedding_model_path = "model_onnx/embedding.onnx"
    onnx_tokenizer_path = "model_onnx/tokenizer.json"
    sentence_transformer_dir = "model/sentence_transformer.model"
    embedding_dim = 384



# @dataclass
# class Settings:
#     s3_bucket_name = "mlops-lab11-models-kaspersky"
#     s3_model_dir = "model/"
#     local_model_dir = "../../model"
#     classifier_joblib_path = local_model_dir + "/classifier.joblib"
#     onnx_classifier_path = "../../model_onnx/classifier.onnx"
#     onnx_embedding_model_path = "../../model_onnx/embedding.onnx"
#     onnx_tokenizer_path = "../../model_onnx/tokenizer.json"
#     sentence_transformer_dir = "../../model/sentence_transformer.model"
#     embedding_dim = 384