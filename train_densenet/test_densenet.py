import torch
import torchvision
import torchextractor as tx
from densenet import DenseNet
model = DenseNet()
# for name, module in model.named_modules():
#     print(name)
feature_names = ["features.relu0", "features.transition1.pool", "features.transition2.pool", "features.transition3.pool", "features.norm5"]
model = tx.Extractor(model, feature_names)
dummy_input = torch.rand(1, 3, 512, 512)
model_output, features = model(dummy_input)

features_list = []

for name in feature_names:
    features_list.append(features[name])

features_list.append(model_output)

for item in features_list:
    print(item.shape)
# print(feature_shapes)
# print(model_output.shape)


# train_nodes, eval_nodes = get_graph_node_names(model)
# print(train_nodes)

# return_nodes = {
#     # node_name: user-specified key for output dict
#     'features.pool0': 'layer1',
#     'features.transition1.pool': 'layer2',
#     'features.transition2.pool': 'layer3',
#     'features.transition3.pool': 'layer4',
# }