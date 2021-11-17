import os
import torch
import torch.multiprocessing as mp
from dataset.skeleton_multitask import *
from utils.evaluation import MultiAttrDatasetEvaluator
from utils.misc import get_folders_and_files


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
mp.set_start_method('spawn', force=True)


def benchmark(evaluator: MultiAttrDatasetEvaluator, m_path: str, o_path: str):
    evaluator.set_model(m_path, o_path)
    evaluator.run_evaluation()


def kill_all_processes(p_list: list):
    for process in p_list:
        process.terminate()


def chunks(l: list, n: int):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':
    use_gpu = False     # torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    dataset_path = './dataset/JL_Dataset_translated_crossSample12.skeldat'
    output_path = ('./validation/'
                   '/')
    cde = MultiAttrDatasetEvaluator(dataset_path=dataset_path, device=device)
    if isinstance(cde.dataset, SkeletonDatasetK3Da):
        if cde.dataset.downsample_factor != 2:
            cde.dataset.downsample_factor = 2
    models_for_testing = get_folders_and_files(output_path)[1]
    processes = []
    for model in models_for_testing:
        if not model.endswith('.tar'):
            continue
        model_path = output_path + model
        p = mp.Process(name=model, target=benchmark, args=(cde, model_path, output_path))
        p.daemon = True     # set false if not running in interactive mode
        p.start()
        processes.append(p)
