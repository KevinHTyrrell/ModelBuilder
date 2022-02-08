import argparse
import os
import yaml
from tensorflow import keras
from Framework.model_builder import ModelBuilder

if __name__ == '__main__':
    model_builder = ModelBuilder()
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file',    type=str,   required=True, help='Model configuration yaml file')
    parser.add_argument('--input_dims',     type=tuple, required=True, help='Input dimensions for model')
    parser.add_argument('--output_dims',    type=int,   required=True, help='Output dimensions for model')
    parser.add_argument('--optimizer',      type=str,   required=True, help='Optimizer for model')
    parser.add_argument('--loss',           type=str,   required=True, help='Loss function for model')
    parser.add_argument('--name',           type=str,   required=True, help='Filename for model')

    args = parser.parse_args()
    model_config_file   = args.config_file
    input_dims          = args.input_dims
    output_dims         = args.output_dims
    optimizer           = args.optimizer
    loss_fn             = args.loss
    name                = args.name

    with open(model_config_file, 'r') as f:
        yaml_contents = yaml.safe_load(f)
    model_config = yaml_contents['Model']

    input_layer, current_layer, build_layer_list = model_builder.build_tensor_model(input_dims=input_dims, model_config=model_config)
    output_layer = keras.layers.Dense(output_dims, activation='softmax')(current_layer)
    model = keras.models.Model(input_layer, output_layer)
    model.compile(optimizer=optimizer, loss=loss_fn)
    outpath = os.path.join('Models', name)
    model.save(outpath)