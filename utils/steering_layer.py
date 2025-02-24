# The following section is adapted from https://github.com/DLR-SC/style-vectors-for-steering-llms
from utils.logging import log_error
from model.model import project_model
import numpy as np
import torch as t

t.set_default_dtype(t.float32)

class SteeringLayer(t.nn.Module):
    """Our custom steering layer for an LLM."""
    
    def __init__(self, target_layer, layer_num, steering_vectors, b):
        super().__init__()
        self.target_layer = target_layer
        self.add_steering = True
        self.layer_num = layer_num
        self.b = t.tensor(b, dtype=t.float32) if not isinstance(b, t.Tensor) else b.to(dtype=t.float32)
        self.steering_vector = self.load_steering_vector(steering_vectors)

    def get_layer_device(self):
        try:
            layer_num = self.layer_num
            device_map = project_model.get_device_map()  
            return device_map[f'model.layers.{layer_num}']
        except Exception as e:
            log_error(f"Error in get_layer_device: {e}! Steering vector is randomly initialized!")
            return None


    def load_steering_vector(self, steering_vectors):
        device = self.get_layer_device()
        try:
            if isinstance(steering_vectors, list):
                # If it's a list, convert each element to a tensor if it's a numpy array
                tensor_list = [t.tensor(v, device=device) if isinstance(v, np.ndarray) else v for v in steering_vectors]
                stacked_vectors = t.stack(tensor_list)
                return t.nn.Parameter(stacked_vectors)
            elif isinstance(steering_vectors, np.ndarray):
                # If it's a numpy array, convert it to a tensor
                return t.nn.Parameter(t.tensor(steering_vectors, device=device))
            elif isinstance(self.steering_vectors, t.Tensor):
                # If it's already a tensor, just move it to the correct device
                return t.nn.Parameter(steering_vectors.to(device))
            else:
                raise TypeError("steering_vectors must be a list of arrays/tensors, a numpy array, or a tensor")
        except Exception as e:
            log_error(f"Error in loading_steering_vector: {e}! Steering vector is randomly initialized!")
            target_layer_dim = project_model.get_config().hidden_size
            return t.nn.Parameter(t.nn.init.xavier_normal_(t.empty((1, target_layer_dim), device=device, dtype=t.float32)))

    def forward(self, *args, **kwargs):
        try:
            original_output = self.target_layer(*args, **kwargs)

            # import pdb; pdb.set_trace()

            if self.add_steering:
                if isinstance(original_output, tuple):
                    # Scale the steering vector to match the scale of original_output
                    first_element = original_output[0].clone()
                    
                    last_token_output = original_output[0][:,-1,:]
                    scaled_steering_vector = self.steering_vector * (last_token_output.abs().mean() / self.steering_vector.abs().mean())
                    
                    first_element[:,-1,:] = last_token_output * (1-self.b) + self.b * scaled_steering_vector.to(dtype=t.float32)
                    
                    return (first_element,) + original_output[1:]
                else:
                    # This part remains unchanged
                    last_token_output = original_output[:,-1,:]
                    scaled_steering_vector = self.steering_vector * (last_token_output.abs().mean() / self.steering_vector.abs().mean())
                    return original_output.to(dtype=t.float32) * (1-self.b) + self.b * scaled_steering_vector.to(dtype=t.float32)
            
            return original_output
        except Exception as e:
            log_error(f"Error in SteeringLayer forward pass: {e}! Returning original layer output.")
            return self.target_layer(*args, **kwargs)


