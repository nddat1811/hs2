import sys
import argparse
import ast
import pickle
from data_utils import SERDataset
import torch
import numpy as np
# from model import SER_AlexNet, SER_AlexNet_GAP, SER_CNN
from models.ser_model import Ser_Model
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import os
import random
from tqdm import tqdm
from collections import Counter
from torch.backends import cudnn

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def write_log(log_path, message):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fout:
        fout.write(message + "\n")


def save_best_checkpoint(previous_best_path, checkpoint, checkpoint_dir, save_label):
    os.makedirs(checkpoint_dir, exist_ok=True)

    epoch = checkpoint["epoch"]
    score = checkpoint["val_score"]
    val_id = checkpoint["params"]["val_id"]
    test_id = checkpoint["params"]["test_id"]
    repeat_idx = checkpoint["params"]["repeat_idx"]
    checkpoint_name = (
        f"{save_label}_repeat{repeat_idx}_val{val_id}_test{test_id}"
        f"_epoch{epoch + 1:03d}_score{score:.4f}.pth"
    )
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)

    if (
        previous_best_path is not None
        and previous_best_path != checkpoint_path
        and os.path.exists(previous_best_path)
    ):
        os.remove(previous_best_path)

    return checkpoint_path


def get_checkpoint_prefix(save_label, params):
    return (
        f"{save_label}_repeat{params['repeat_idx']}"
        f"_val{params['val_id']}_test{params['test_id']}"
    )


def get_latest_checkpoint_path(checkpoint_dir, save_label, params):
    prefix = get_checkpoint_prefix(save_label, params)
    return os.path.join(checkpoint_dir, f"{prefix}_latest.pth")


def save_latest_checkpoint(checkpoint, checkpoint_dir, save_label):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = get_latest_checkpoint_path(
        checkpoint_dir,
        save_label,
        checkpoint["params"],
    )
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def find_best_checkpoint(checkpoint_dir, save_label, params):
    if not os.path.isdir(checkpoint_dir):
        return None

    prefix = get_checkpoint_prefix(save_label, params)
    best_checkpoint = None
    for file_name in os.listdir(checkpoint_dir):
        if not file_name.startswith(prefix):
            continue
        if not file_name.endswith(".pth") or file_name.endswith("_latest.pth"):
            continue
        checkpoint_path = os.path.join(checkpoint_dir, file_name)
        score = -1e8
        if "_score" in file_name:
            score_text = file_name.rsplit("_score", 1)[1].removesuffix(".pth")
            try:
                score = float(score_text)
            except ValueError:
                score = -1e8
        item = {
            "path": checkpoint_path,
            "score": score,
            "epoch": 0,
        }
        if best_checkpoint is None or item["score"] > best_checkpoint["score"]:
            best_checkpoint = item

    return best_checkpoint["path"] if best_checkpoint is not None else None


def remove_extra_best_checkpoints(checkpoint_dir, save_label, params, keep_path):
    if keep_path is None or not os.path.isdir(checkpoint_dir):
        return

    prefix = get_checkpoint_prefix(save_label, params)
    for file_name in os.listdir(checkpoint_dir):
        if not file_name.startswith(prefix):
            continue
        if not file_name.endswith(".pth") or file_name.endswith("_latest.pth"):
            continue
        checkpoint_path = os.path.join(checkpoint_dir, file_name)
        if checkpoint_path != keep_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def load_training_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def parse_config_value(value):
    value = value.strip()
    if not value:
        return None

    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none"):
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("\"'")


def load_config(config_path):
    if config_path is None or not os.path.exists(config_path):
        return {}

    config = {}
    with open(config_path, "r", encoding="utf-8") as fin:
        for line_number, raw_line in enumerate(fin, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Invalid config line {line_number}: {raw_line.rstrip()}")
            key, value = line.split(":", 1)
            config[key.strip()] = parse_config_value(value)

    return config


colors_per_class = {
    0 : [0, 0, 0],
    1 : [255, 107, 107],
    2 : [100, 100, 255],
    3 : [16, 172, 132],
}
def main(args):
    
    # Aggregate parameters
    params={
            #model & features parameters
            'ser_task': 'SLM',

            #training
            'repeat_idx': args.repeat_idx,
            'val_id': args.val_id,
            'test_id': args.test_id,
            'num_epochs':args.num_epochs,
            'early_stop':args.early_stop,
            'batch_size':args.batch_size,
            'lr':args.lr,
            'random_seed':args.seed,
            'use_gpu':args.gpu,
            'gpu_ids': args.gpu_ids,
            'dataset_type':args.dataset_type,
            'wavlm_path': args.wavlm_path,
            'checkpoint_dir': args.checkpoint_dir,
            'log_file': args.log_file,
            'config': args.config,
            'resume': args.resume,
             
            #best mode
            'save_label': args.save_label,
            #parameters for tuning
            'oversampling': args.oversampling,
            'pretrained': args.pretrained
            }

    print('*'*40)
    print(f"\nPARAMETERS:\n")
    print('*'*40)
    print('\n')
    for key in params:
        print(f'{key:>15}: {params[key]}')
    print('*'*40)
    print('\n')

    #set random seed
    seed_everything(params['random_seed'])
    # Load dataset
    with open(args.features_file, "rb") as fin:
        features_data = pickle.load(fin)
    ser_dataset = SERDataset(features_data,
                               val_speaker_id=args.val_id,
                               test_speaker_id=args.test_id,
                               oversample=args.oversampling,
                               dataset_type=args.dataset_type
                               )
    # Train
    train_stat = train(ser_dataset, params, args, save_label=args.save_label)

    return train_stat


def parse_arguments(argv):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default='config.yaml',
        help='Path to the training config file.')
    config_args, remaining_argv = config_parser.parse_known_args(argv)
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a SER  model in an iterative-based manner with "
                    "pyTorch and IEMOCAP dataset.")

    #Features
    parser.add_argument('features_file', type=str, nargs='?',
        default=config.get('features_file'),
        help='Features extracted from `extract_features.py`.')
     
    #Training
    parser.add_argument('--repeat_idx', type=str, default=config.get('repeat_idx', '0'),
        help='ID of repeat_idx')
    parser.add_argument('--val_id', type=str, default=config.get('val_id', '1F'),
        help='ID of speaker to be used as validation')
    parser.add_argument('--test_id', type=str, default=config.get('test_id', '1M'),
        help='ID of speaker to be used as test')
    parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs', 200),
        help='Number of training epochs.') 
    parser.add_argument('--early_stop', type=int, default=config.get('early_stop', 4),
        help='Number of early stopping epochs.') 
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 8),
        help='Mini batch size.')
    parser.add_argument('--lr', type=float, default=config.get('lr', 0.0001), 
        help='Learning rate.')
    parser.add_argument('--seed', type=int, default=config.get('seed', 100),
        help='Random seed for reproducibility.')
    parser.add_argument('--gpu', type=int, default=config.get('gpu', 1),
        help='If 1, use GPU')
    parser.add_argument('--gpu_ids', default=config.get('gpu_ids', [0]),
        help='If 1, use GPU')
    parser.add_argument('--dataset_type', type=str, default=config.get('dataset_type', "IEMOCAP"),
        help="IEMOCAP" 
              "EMODB"
              "RAVDESS"
              "MELD")     
    parser.add_argument('--wavlm_path', type=str, default=config.get('wavlm_path', "facebook/wav2vec2-base-960h"),
        help='Path or Hugging Face model id for WavLM-large.')
     
    #Best Model
    parser.add_argument('--save_label', type=str, default=config.get('save_label'),
        help='Label for the current run, used to save the best model ')
    parser.add_argument('--checkpoint_dir', type=str, default=config.get('checkpoint_dir', 'checkpoints'),
        help='Directory used to save the top checkpoint files.')
    parser.add_argument('--log_file', type=str, default=config.get('log_file'),
        help='Training log file. If not set, <save_label>.log is used.')
    parser.add_argument('--resume', action='store_true', default=config.get('resume', True),
        help='Resume from the latest checkpoint for this run if it exists.')
    parser.add_argument('--no_resume', action='store_false', dest='resume',
        help='Start training from epoch 1 even if a latest checkpoint exists.')

    #Parameters for model tuning
    parser.add_argument('--oversampling', action='store_true', default=config.get('oversampling', False),
        help='By default, no oversampling is applied to training dataset.'
             'Set this to true to apply random oversampling to balance training dataset')
     
    parser.add_argument('--pretrained', action='store_true', default=config.get('pretrained', False),
        help='By default, SER_AlexNet or SER_AlexNet_GAP model weights are'
             'initialized randomly. Set this flag to initalize with '
             'ImageNet pre-trained weights.')

    args = parser.parse_args(remaining_argv)
    args.config = config_args.config
    if args.features_file is None:
        parser.error("features_file is required. Set it in config.yaml or pass it as an argument.")

    return args



def test(mode, params, args, model, criterion_ce, criterion_mml, test_dataset, batch_size, device,
         dataset_type=None,
         return_matrix=False):
    #赋值dataset_type.
    if dataset_type is None:
        dataset_type = args.dataset_type

    """Test an SER model.

    Parameters
    ----------
    model
        PyTorch model
    criterion
        loss_function
    test_dataset
        The test dataset
    batch_size : int
    device
    return_matrix : bool
        Whether to return the confusion matrix.

    Returns
    -------
    loss, weighted accuracy (WA), unweighted accuracy (UA), confusion matrix 
       

    """
    total_loss = 0
    test_preds_segs = []
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # we'll store the features as NumPy array of size num_images x feature_size and the labels
    sne_features1 = None
    sne_features2 = None
    sne_features3 = None
    sne_features4 = None
    
    sne_features5 = None
    sne_features6 = None
    sne_features7 = None
    sne_features8 = None
    
    sne_features9 = None
    sne_features10 = None
    sne_features11 = None
    sne_features12 = None
    
    sne_features13 = None
    sne_features14 = None
    sne_features15 = None
    sne_features16 = None
    
    sne_features17 = None
    sne_features18 = None
    sne_features19 = None
    sne_features20 = None
    
    # sne_features = [sne_features1, sne_features2, sne_features3, sne_features4, sne_features5, sne_features6, sne_features7, sne_features8, sne_features9, sne_features10, sne_features11, sne_features12, sne_features13, sne_features14, sne_features15, sne_features16, sne_features17, sne_features18, sne_features19, sne_features20]
    
    sne_features = [None, None, None, None]
    out_features = None
    sne_labels = []
        
    model.eval()

    # for i, test_batch in enumerate(test_loader):
    with tqdm(test_loader) as td:
        for test_batch in td:
                
            # Send data to correct device
            test_data_spec_batch = test_batch['seg_spec'].to(device)
            test_data_mfcc_batch = test_batch['seg_mfcc'].to(device)
            test_data_audio_batch = test_batch['seg_audio'].to(device)
            test_labels_batch =  test_batch['seg_label'].to(device,dtype=torch.long)
        
            labels = test_batch['seg_label'].cpu().detach().numpy()
            sne_labels += list(labels)
        
            # Forward
            test_outputs = model(test_data_spec_batch, test_data_mfcc_batch, test_data_audio_batch)
            test_preds_segs.append(f.log_softmax(test_outputs['M'], dim=1).cpu())

            #test loss
            test_loss_ce = criterion_ce(test_outputs['M'], test_labels_batch)
            # test_loss_mml = criterion_mml(test_outputs['M'], test_labels_batch)
            test_loss = test_loss_ce#  + test_loss_mml
           
            total_loss += test_loss.item()
            
            '''
            # VISULAIZATION
            for index in range(4):
                str_idx = 'F' + str(index+1)
                current_features = test_outputs[str_idx].cpu().numpy()
                if sne_features[index] is not None:
                    sne_features[index] = np.concatenate((sne_features[index], current_features))
                else:
                    sne_features[index] = current_features     
            '''           
    '''
    # VISULAIZATION
    if mode == 'TEST':
        for index in range(4):
            tsne = TSNE(n_components=2).fit_transform(sne_features[index])
            visualize_tsne_2(str(index), tsne, sne_labels, params)    
    '''
    # Average loss
    test_loss = total_loss / len(test_loader)

    # Accumulate results for val data
    test_preds_segs = np.vstack(test_preds_segs)
    test_preds = test_dataset.get_preds(test_preds_segs)
    
    # Make sure everything works properly
    assert len(test_preds) == test_dataset.n_actual_samples
    test_wa = test_dataset.weighted_accuracy(test_preds)
    test_ua = test_dataset.unweighted_accuracy(test_preds)
    test_cor = test_dataset.confusion_matrix(test_preds,dataset_type)

    test_f1 = test_dataset.weighted_f1(test_preds)     # ★ 新增

    results = (test_loss, test_wa*100, test_ua*100, test_f1*100)   # ★ 修改为返回 WF1
    
    if return_matrix:
        test_conf = test_dataset.confusion_matrix(test_preds,dataset_type)
        return results, test_conf
    else:
        return results

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
    
def visualize_tsne_points_2(name, tx, ty, labels, params):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, s=1, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.show()
    
    t = round(time.time()*1000)
    t_str = time.strftime('%H_%M_%S',time.localtime(t/1000))

    img_path = './results/t-SNE/' + t_str + '_' + name + '_' + params['repeat_idx'] + '_' + params['test_id'] + '.png'
    # finally, show the plot
    fig.savefig(img_path, dpi=fig.dpi)
    
def visualize_tsne_points_3(name, tx, ty, tz, labels, params):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        current_tz = np.take(tz, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, current_tz, s=4, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    
    t = round(time.time()*1000)
    t_str = time.strftime('%H_%M_%S',time.localtime(t/1000))

    img_path = './results/t-SNE/' + t_str + '_' + name + '_' + params['repeat_idx'] + '_' + params['test_id'] + '.png'
    print(img_path)
    # finally, show the plot
    fig.savefig(img_path, dpi=fig.dpi)
        
def visualize_tsne_2(name, tsne, labels, params):

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points_2(name, tx, ty, labels, params)

def visualize_tsne_3(name, tsne, labels, params):

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tz = tsne[:, 2]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    tz = scale_to_01_range(tz)

    # visualize the plot: samples as colored points
    visualize_tsne_points_3(name, tx, ty, tz, labels, params)
    
def train(dataset, params, args, save_label='default'):
    if save_label is None:
        save_label = time.strftime("%Y-%m-%d-%H-%M-%S")
        params["save_label"] = save_label

    #get dataset
    train_dataset = dataset.get_train_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=params['batch_size'], 
                                shuffle=True)  
                                
    val_dataset = dataset.get_val_dataset()
    test_dataset = dataset.get_test_dataset()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("pytorch version: ", torch.__version__)
    print("cuda version: ", torch.version.cuda)
    print("cudnn version: ", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        print("gpu name: ", torch.cuda.get_device_name())
        print("gpu index: ", torch.cuda.current_device())
    else:
        print("gpu name: CPU only")
    
    #select device
    if params['use_gpu'] == 1 and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Construct model, optimizer and criterion
    batch_size = params['batch_size']
    
    # print(type(Ser_Model()))
    model = Ser_Model(num_classes=dataset.num_classes, wavlm_path=params['wavlm_path']).to(device) 
    
    print(model.eval())
    print(f"Number of trainable parameters: {count_parameters(model.train())}")
    print('\n')

    #Set loss criterion and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mml = nn.MultiMarginLoss(margin=0.5) 

    loss_format = "{:.04f}"
    acc_format = "{:.02f}%"
    acc_format2 = "{:.02f}"
    best_val_wa = 0
    best_val_ua = 0
    best_val_f1 = 0
    checkpoint_dir = params['checkpoint_dir']
    log_path = params['log_file'] if params['log_file'] is not None else save_label + '.log'
    best_checkpoint_path = None
    best_val_loss = 1e8
    best_val_acc = -1e8
    best_epoch = 0
    start_epoch = 0
    run_start_time = time.time()

    latest_checkpoint_path = get_latest_checkpoint_path(checkpoint_dir, save_label, params)
    best_checkpoint_path = None
    if params['resume'] and os.path.exists(latest_checkpoint_path):
        best_checkpoint_path = find_best_checkpoint(checkpoint_dir, save_label, params)
        remove_extra_best_checkpoints(checkpoint_dir, save_label, params, best_checkpoint_path)
        latest_checkpoint = load_training_checkpoint(latest_checkpoint_path, device)
        if "model_state_dict" in latest_checkpoint:
            model.load_state_dict(latest_checkpoint["model_state_dict"])
        else:
            model.load_state_dict(latest_checkpoint)
        if "optimizer_state_dict" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer_state_dict"])

        start_epoch = latest_checkpoint.get("epoch", -1) + 1
        best_epoch = latest_checkpoint.get("best_epoch", latest_checkpoint.get("epoch", 0))
        best_val_loss = latest_checkpoint.get("best_val_loss", best_val_loss)
        best_val_wa = latest_checkpoint.get("best_val_wa", best_val_wa)
        best_val_ua = latest_checkpoint.get("best_val_ua", best_val_ua)
        best_val_f1 = latest_checkpoint.get("best_val_f1", best_val_f1)
        best_val_acc = best_val_wa + best_val_ua

    write_log(log_path, "=" * 80)
    write_log(log_path, f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(log_path, f"save_label: {save_label}")
    write_log(log_path, f"checkpoint_dir: {checkpoint_dir}")
    write_log(log_path, f"repeat_idx: {params['repeat_idx']} | val_id: {params['val_id']} | test_id: {params['test_id']}")
    write_log(log_path, f"dataset_type: {params['dataset_type']} | batch_size: {params['batch_size']} | lr: {params['lr']}")
    if start_epoch > 0:
        write_log(log_path, f"Resumed from: {latest_checkpoint_path}")
        write_log(log_path, f"Resume start_epoch: {start_epoch + 1}")
        print(f"Resume from checkpoint: {latest_checkpoint_path}")
        print(f"Continue training from epoch {start_epoch + 1}")
    else:
        write_log(log_path, "Resume checkpoint: not found, start from epoch 1")
    write_log(log_path, "epoch,elapsed,lr,train_loss,train_wa,train_ua,train_wf1,val_loss,val_wa,val_ua,val_wf1,val_score,best_epoch,best_val_wa,best_val_ua,best_val_wf1,checkpoint_saved")
     
    all_train_loss =[]
    all_train_wa =[]
    all_train_ua=[]
    all_val_loss=[]
    all_val_wa=[]
    all_val_ua=[]
    all_val_wf1=[]
    train_preds = []
    all_train_f1 = []
    
    print("Start Training!!!")
    
    if start_epoch >= params['num_epochs']:
        print(f"Checkpoint already reached num_epochs ({params['num_epochs']}).")
        write_log(log_path, f"Checkpoint already reached num_epochs ({params['num_epochs']}).")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, params['num_epochs']):
        
        y_pred = {'M': [], 'A': []}
        y_true = {'M': [], 'A': []}
        
        # adjust_learning_rate(params['lr'], optimizer, epoch)
        
        #get current learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        
        # Train one epoch
        total_loss = 0
        train_preds = []
        target=[]
        model.train()
        
        # for i, train_batch in enumerate(train_loader):
        with tqdm(train_loader) as td:
            for train_batch in td:
                
                # Clear gradients
                optimizer.zero_grad()
            
                # Send data to correct device
                train_data_spec_batch = train_batch['seg_spec'].to(device)
                train_data_mfcc_batch = train_batch['seg_mfcc'].to(device)
                train_data_audio_batch = train_batch['seg_audio'].to(device)
                train_labels_batch =  train_batch['seg_label'].to(device,dtype=torch.long)
            
                # Forward pass
                outputs = model(train_data_spec_batch, train_data_mfcc_batch, train_data_audio_batch)
            
                #for m in params['ser_task']:
                #    y_pred[m].append(f.log_softmax(outputs[m], dim=1).cpu().detach().numpy())
                #    y_true.append
                
                train_preds.append(f.log_softmax(outputs['M'], dim=1).cpu().detach().numpy())
                
                # Compute the loss, gradients, and update the parameters
                train_loss_ce = criterion_ce(outputs['M'], train_labels_batch)
                # train_loss_mml = criterion_mml(outputs['M'], train_labels_batch)
                train_loss = train_loss_ce# + train_loss_mml

                train_loss.backward()
                total_loss += train_loss.item()
                optimizer.step()
            
        # Evaluate training data
        train_loss = total_loss / len(train_loader)
        # Accumulate results for train data
        train_preds = np.vstack(train_preds)
        train_preds = train_dataset.get_preds(train_preds)
        
        # Make sure everything works properly
        train_wa = train_dataset.weighted_accuracy(train_preds) * 100
        train_ua = train_dataset.unweighted_accuracy(train_preds) * 100
        train_f1 = train_dataset.weighted_f1(train_preds) * 100
        #train_cor = train_dataset.confusion_matrix_iemocap(train_preds)
        
        all_train_loss.append(loss_format.format(train_loss))
        all_train_wa.append(acc_format2.format(train_wa))
        all_train_ua.append(acc_format2.format(train_ua))
        all_train_f1.append(acc_format2.format(train_f1))
    
    
        #Validation
        with torch.no_grad():
            val_result = test('VAL', params, args,
                model, criterion_ce, criterion_mml, val_dataset, 
                batch_size=batch_size, 
                device=device)

            val_loss = val_result[0]
            val_wa = val_result[1]
            val_ua = val_result[2]
            val_f1 = val_result[3]

            # Update best model based on validation UA
            # if val_loss < (best_val_loss - 1e-6):
            #使用IEMOCAP,RAVDESS,EMODB数据集时，
            is_best_epoch = False
            if val_wa + val_ua > best_val_acc:
                 
                print("True")
                is_best_epoch = True
                best_val_ua = val_ua
                best_val_wa = val_wa
                best_val_loss = val_loss
                best_val_f1 = val_f1
                best_val_acc = val_wa + val_ua
                best_epoch = epoch
            #使用MELD数据集时，
            # if val_f1 > (best_val_f1 + 1e-6):  # 增加微小值避免浮点精度问题
            #     print("True (Best F1 updated)")
            #     best_val_ua = val_ua
            #     best_val_wa = val_wa
            #     best_val_loss = val_loss
            #     best_val_f1 = val_f1  # 保存最佳F1值
            #     best_val_acc = val_wa + val_ua  # 保留原有acc计算
            #     best_epoch = epoch
            #     if save_path is not None:
            #         torch.save(model.state_dict(), save_path)
            print(best_epoch, epoch)

        all_val_loss.append(loss_format.format(val_loss))
        all_val_wa.append(acc_format2.format(val_wa))
        all_val_ua.append(acc_format2.format(val_ua))
        all_val_wf1.append(acc_format2.format(val_f1))

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "params": params,
            "train_loss": float(train_loss),
            "train_wa": float(train_wa),
            "train_ua": float(train_ua),
            "train_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_wa": float(val_wa),
            "val_ua": float(val_ua),
            "val_f1": float(val_f1),
            "val_score": float(val_wa + val_ua),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "best_val_wa": float(best_val_wa),
            "best_val_ua": float(best_val_ua),
            "best_val_f1": float(best_val_f1),
        }
        saved_checkpoint_path = None
        if is_best_epoch:
            best_checkpoint_path = save_best_checkpoint(
                best_checkpoint_path,
                checkpoint,
                checkpoint_dir,
                save_label,
            )
            saved_checkpoint_path = best_checkpoint_path
        latest_checkpoint_path = save_latest_checkpoint(
            checkpoint,
            checkpoint_dir,
            save_label,
        )

        elapsed = time.time() - run_start_time
        elapsed_text = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        write_log(
            log_path,
            "{},{},{:.8f},{:.4f},{:.2f},{:.2f},{:.2f},{:.4f},{:.2f},{:.2f},{:.2f},{:.4f},{},{:.2f},{:.2f},{:.2f},{}".format(
                epoch + 1,
                elapsed_text,
                current_lr,
                train_loss,
                train_wa,
                train_ua,
                train_f1,
                val_loss,
                val_wa,
                val_ua,
                val_f1,
                val_wa + val_ua,
                best_epoch + 1,
                best_val_wa,
                best_val_ua,
                best_val_f1,
                saved_checkpoint_path if saved_checkpoint_path is not None else latest_checkpoint_path,
            )
        )
         
        print(f"Epoch {epoch+1}  (lr = {current_lr})\
        Loss: {loss_format.format(train_loss)} - {loss_format.format(val_loss)} - WA: {acc_format.format(val_wa)} <{acc_format.format(best_val_wa)}> - UA: {acc_format.format(val_ua)} <{acc_format.format(best_val_ua)}> - WF1: {acc_format.format(val_f1)} <{acc_format.format(best_val_f1)}>")

        # early stop
        if (epoch - best_epoch >= params['early_stop']) and (epoch > 5):
            break    
        #break
        
    # Test on best model
    with torch.no_grad():
        if best_checkpoint_path is None:
            best_checkpoint_path = latest_checkpoint_path if os.path.exists(latest_checkpoint_path) else None
        if best_checkpoint_path is None:
            raise RuntimeError("No checkpoint was saved during training.")
        best_checkpoint = load_training_checkpoint(best_checkpoint_path, device)
        if "model_state_dict" in best_checkpoint:
            model.load_state_dict(best_checkpoint["model_state_dict"])
        else:
            model.load_state_dict(best_checkpoint)

        test_result, confusion_matrix = test('TEST', params, args,
            model, criterion_ce, criterion_mml, test_dataset, 
            batch_size=batch_size,
            device=device,
            dataset_type=args.dataset_type,
            return_matrix=True)

        print("*" * 40)
        print("RESULTS ON TEST SET:")
        print("Loss:{:.4f}\tWA: {:.2f}\tUA: {:.2f}\tWF1: {:.2f}".format(test_result[0], test_result[1], test_result[2], test_result[3]))

        print("Confusion matrix:\n{}".format(confusion_matrix[1]))   
        write_log(log_path, f"Best checkpoint: {best_checkpoint_path}")
        write_log(log_path, "TEST,Loss:{:.4f},WA:{:.2f},UA:{:.2f},WF1:{:.2f}".format(test_result[0], test_result[1], test_result[2], test_result[3]))
        write_log(log_path, "Run finished: {}".format(time.strftime('%Y-%m-%d %H:%M:%S')))
         

    return(epoch, best_epoch, 
       all_train_loss, all_train_wa, all_train_ua, all_train_f1,
       all_val_loss, all_val_wa, all_val_ua, all_val_wf1,
       loss_format.format(test_result[0]), 
       acc_format2.format(test_result[1]),
       acc_format2.format(test_result[2]),
       acc_format2.format(test_result[3]),
       confusion_matrix[0])


# seeding function for reproducibility
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #cudnn.benchmark=True
    #cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(lr_0, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_0 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
# to count the number of trainable parameter in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
