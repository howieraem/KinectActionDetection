"""Visualize prediction results of samples from test sets."""
import torch
from utils.misc import load_checkpoint
from dataset.skeleton_abstract import deserialize_dataset
from utils.visualizer import visualize_result


if __name__ == '__main__':
    use_gpu = False     # torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    original_dataset = deserialize_dataset('./dataset/'
                                           'OAD_dataset'
                                           '.skeldat', False)
    has_interaction = original_dataset.has_interaction
    input_dim = original_dataset.get_joint_number() * 3
    if has_interaction:
        input_dim *= 2
    model = load_checkpoint('./validation/'
                            'OAD_VA+LN+SRU'
                            '.tar',
                            num_classes=original_dataset.label_size,
                            input_dim=input_dim,
                            device=device)[0]
    visualize_result(original_dataset, model, device, fps=30)
