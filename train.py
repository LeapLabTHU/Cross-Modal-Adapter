from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import pdb
import shutil

import torch
import numpy as np
import random
import os

from tqdm import tqdm

if 'ceph-jd' in os.getcwd():
    HF = True
    import hfai_env

    hfai_env.set_env('nzl_py38')
else:
    HF = False

from pathlib import Path

from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_jointvt_upshare_adapter import CLIP4Clip
from modules.optimization import BertAdam

from util import parallel_apply, get_logger

from handler import ExpHandler
from comm import is_main_process
import torch.multiprocessing as mp
import torch.distributed as dist

global logger
run_obj = None


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", type=int, default=1)
    parser.add_argument("--do_eval", type=int, default=0)

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    parser.add_argument('--anno_path', type=str, default='/home/data/MSR-VTT/anns',
                        help='annotation path')
    parser.add_argument("--train7k", action='store_true')

    parser.add_argument('--workers', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight_decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--train_epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=16, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--max_frames', type=int, default=12, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', type=int, default=1)

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=2, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf", "xpool"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    parser.add_argument('--en_wandb', type=int, default=0, choices=[0, 1])
    parser.add_argument('--debug', type=int, default=0, choices=[0, 1])
    parser.add_argument('--no_resume', type=int, default=0, choices=[0, 1])

    parser.add_argument('--jcl', type=float, default=0)
    parser.add_argument('--sigma_lambda', type=float, default=1)

    parser.add_argument('--hf', type=int, default=0)
    parser.add_argument('--vsim_temp', type=float, default=0.1)

    parser.add_argument('--intra_sim', type=float, default=0)
    parser.add_argument('--text_cond_intra_sim', type=float, default=0)
    parser.add_argument('--intra_sim_logit_scale', type=float, default=0)
    parser.add_argument('--mq_test', type=int, default=0)

    parser.add_argument('--no_expand_type', type=str, default='all', choices=['all', 'first', 'rand'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--schedule', type=str, default='warmup_cosine', choices=['warmup_cosine', 'warmup_constant', 'warmup_linear'])
    parser.add_argument('--clip_lr', type=float, default=1e-7)
    parser.add_argument('--linspace_samp', type=str, default=None, choices=['rand', 'uniform'])
    parser.add_argument('--q_score', type=int, default=1)
    parser.add_argument('--q_score_proj', type=int, default=0)
    parser.add_argument('--q_score_temp', type=float, default=5)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)

    parser.add_argument("--v_emb_norm", type=int, default=0,
                        help="Whether to normalize visual embedding before calculating qscore")
    parser.add_argument("--t_emb_norm", type=int, default=0,
                        help="Whether to normalize textual embedding before calculating qscore")
    parser.add_argument("--v_agg_norm", type=int, default=0,
                        help="Whether to normalize aggregated video representation during qscore calculating")

    parser.add_argument("--dataset_type", type=str, default='drl', choices=['ori', 'drl', 'img'])

    ### Adapter
    parser.add_argument("--adapter_reduction_dims", type=int, default=8, help="neck dimension of adapter")
    parser.add_argument("--adapter_share_dim", type=int, default=0, help="shared hypernet dimension")
    parser.add_argument("--adapter_non_linearity", type=str, default='gelu', help="non linearity function of adapter")
    parser.add_argument("--adapter_share_use_non_linearity", action='store_true')
    parser.add_argument("--adapter_add_layer_norm_before_adapter", action='store_true')
    parser.add_argument("--adapter_add_layer_norm_after_adapter", action='store_true')
    parser.add_argument('--adapter_init_std', type=float, default=1e-2, help="adapter nn.Linear or nn.Embedding weight normal init std")
    parser.add_argument("--adapter_convert2fp16", action='store_true')
    parser.add_argument("--adapter_dropout", type=float, default=0.)
    ### End

    parser.add_argument("--train_augment", action='store_true', help="whether use strong traing augmentations")
    parser.add_argument("--horizontal_flip", action='store_true', help="whether use horizontal flip augmentations")

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def set_seed_logger(args):

    assert args.warmup_proportion == 0.1
    assert args.adapter_non_linearity == 'quick_gelu'
    assert args.seed == 42
    assert args.slice_framepos == 2
    assert args.weight_decay == 0.2
    
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    gpu_device_type = torch.cuda.get_device_name(0).replace(' ', '-')
    torch_version = torch.__version__

    exp_name = gpu_device_type + '_torch' + torch_version + '_' \
               + ('qscore{}_'.format(args.q_score_temp) if args.q_score else 'meanP_') \
               + '{}_'.format(args.pretrained_clip_name.replace('/', '-')) \
               + 'seed{}_'.format(args.seed) \
               + 'sliceframe{}'.format(args.slice_framepos) \
               + '___e{}-'.format(args.epochs) \
               + 'bs{}-'.format(args.batch_size) \
               + 'gas{}-'.format(args.gradient_accumulation_steps) \
               + 'lr{}-'.format(args.lr) \
               + 'sched{}-'.format(args.schedule.replace('_', '-')) \
               + 'prop{}-'.format(args.warmup_proportion) \
               + 'opt{}-'.format(args.optimizer) \
               + 'wd{}-'.format(args.weight_decay) \
               + 'gn{}-'.format(args.clip_grad_norm) \
               + 'wks{}'.format(args.workers) \
               + ('_aug' if args.train_augment else '') \
               + ('-hflip' if args.horizontal_flip else '') \
               + '___' \
               + 'sharedim{}-'.format(args.adapter_share_dim) \
               + 'neck{}-'.format(args.adapter_reduction_dims) \
               + '{}-'.format(args.adapter_non_linearity) \
               + ('sharewithnonlinear-' if args.adapter_share_use_non_linearity else '') \
               + ('dpr{}-'.format(args.adapter_dropout) if args.adapter_dropout > 0 else '') \
               + ('fp16-'.format(args.adapter_convert2fp16) if args.adapter_convert2fp16 else '') \
               + 'std{}'.format(args.adapter_init_std)

    args.output_dir = os.path.join(args.output_dir, exp_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                      task_config=args)

    model.to(device)

    return model


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_adapter_param_tp = [(n, p) for n, p in param_optimizer if "adapter" in n and 'bias' not in n]
    no_decay_adapter_param_tp = [(n, p) for n, p in param_optimizer if "adapter" in n and 'bias' in n]

    for n, p in decay_adapter_param_tp:
        print('decay_adapter_param_tp', n)

    for n, p in no_decay_adapter_param_tp:
        print('no_decay_adapter_param_tp', n)

    weight_decay = args.weight_decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_adapter_param_tp], 'weight_decay': weight_decay, 'lr': args.lr},
        {'params': [p for n, p in no_decay_adapter_param_tp], 'weight_decay': 0.0, 'lr': args.lr},
    ]

    total_parameters = [(n, p) for n, p in param_optimizer if "clip." in n]
    only_clip_parameters = [(n, p) for n, p in param_optimizer if "clip." in n and "adapter" not in n]
    adapter_parameters = [(n, p) for n, p in param_optimizer if "clip." in n and "adapter" in n]

    if args.local_rank == 0:
        logger.info('Total parameters = {:.4f} M'.format(sum([np.prod(p.size()) for n, p in total_parameters]) / 1e6))
        logger.info(
            'Only clip arameters = {:.6f} M'.format(sum([np.prod(p.size()) for n, p in only_clip_parameters]) / 1e6))
        logger.info(
            'Adapter parameters = {:.6f} M'.format(sum([np.prod(p.size()) for n, p in adapter_parameters]) / 1e6))
        logger.info('Percent of finetuned parameters = {:.4f} %'.format(
                sum([np.prod(p.size()) for n, p in adapter_parameters]) / sum(
                    [np.prod(p.size()) for n, p in total_parameters]) * 100))

    scheduler = None
    if args.optimizer == 'adam':
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                             schedule=args.schedule, b1=0.9, b2=0.98, e=1e-6,
                             t_total=num_train_optimization_steps, weight_decay=weight_decay,
                             max_grad_norm=0.0)
    elif args.optimizer == 'adamw':
        from modules.optimization_xpool import AdamW, get_cosine_schedule_with_warmup
        optimizer = AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(
                                                        num_train_optimization_steps * args.warmup_proportion),
                                                    num_training_steps=num_train_optimization_steps)
    else:
        raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank)

    return optimizer, scheduler, model, model.module


def save_model(epoch, args, model, optimizer, tr_loss, type_name="", ckpt_path="", scheduler=None):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    if is_main_process():
        torch.save({
            'model': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch,
            'args': args,
        }, ckpt_path)


def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                          task_config=args)

        model.to(device)
    else:
        model = None
    return model


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    data_start_time = time.time()
    total_data_time = 0
    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) if not isinstance(t, tuple) else t for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, video_id = batch
        if args.jcl:
            ratio = args.sigma_lambda * ((global_step + 1) * 1.0 / (len(train_dataloader) * args.epochs))
        else:
            ratio = None
        loss = model(input_ids, segment_ids, input_mask, video, video_mask, sigma_ratio=ratio)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %.10f, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            float(optimizer.get_lr()[0]) if scheduler is None else float(scheduler.get_lr()[0]),
                            # "-".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                run_obj.write(epoch=epoch + 1,
                              max_step=int(len(train_dataloader) * args.epochs),
                              train_metrics=dict(
                                  step=step + 1,
                                  loss=float(loss),
                                  time_per_step=(time.time() - start_time) / (
                                          log_step * args.gradient_accumulation_steps),
                                  lr=optimizer.get_lr()[0] if scheduler is None else scheduler.get_lr()[0]
                              )
                              )
                start_time = time.time()
        # data_start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
                                                             loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix


def eval_epoch(args, model, test_dataloader, device, n_gpu):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(tqdm(test_dataloader)):
            batch = tuple(t.to(device=device) if not isinstance(t, tuple) else t for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask, video_id = batch

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]
                # pdb.set_trace()

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    # print(video.shape)
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids,
                                                                                  input_mask, video, video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list,
                                        batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

        if multi_sentence_:
            logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
            cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
            max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
            sim_matrix_new = []
            for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
                sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                      np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                                                     axis=0))
            sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
            logger.info("after reshape, sim matrix size: {} x {} x {}".
                        format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

            tv_metrics = tensor_text_to_video_metrics(sim_matrix)
            vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
        else:
            logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
            if sim_matrix.shape[0] != sim_matrix.shape[1]:
                sim_matrix = sim_matrix.reshape(sim_matrix.shape[1], 20, sim_matrix.shape[1]).mean(axis=1)
            tv_metrics = compute_metrics(sim_matrix)
            vt_metrics = compute_metrics(sim_matrix.T)
            logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
    if is_main_process():
        run_obj.write(
            eval_metrics={**{f'tv_{k}': v for k, v in tv_metrics.items()},
                          **{f'vt_{k}': v for k, v in vt_metrics.items()}}
        )

    R1 = tv_metrics['R1']
    return R1


def main(local_rank, args):
    global logger, run_obj
    if HF:
        args.local_rank = local_rank
        ip = os.environ['MASTER_IP']
        port = os.environ['MASTER_PORT']
        hosts = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpus = torch.cuda.device_count()
        dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts * gpus,
                                rank=rank * gpus + local_rank)
    else:
        dist.init_process_group(backend="nccl")
    dist.barrier()

    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)
    if is_main_process():
        run_obj = ExpHandler(args)

    tokenizer = ClipTokenizer()

    assert args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    if args.local_rank == 0:
        logger.info(str(model))

    ## ####################################
    # freeze testing
    ## ####################################
    assert 12 >= args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("adapter") != -1:
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    for name, param in model.clip.named_parameters():
        if param.requires_grad:
            print('{} require gradient: {}'.format(name, param.requires_grad))

    ## ####################################
    # dataloader loading
    ## ####################################
    if args.dataset_type == 'ori':
        from dataloaders.data_dataloaders import DATALOADER_DICT
    else:
        from dataloaders_drl.data_dataloaders import DATALOADER_DICT

    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    if args.train7k:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train7k"](args, tokenizer)
    else:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs

    optimizer, scheduler, model, model_without_ddp = prep_optimizer(args, model, num_train_optimization_steps,
                                                                    args.local_rank)

    start_epoch = 0
    global_step = 0
    ## ##############################################################
    # resume optimizer state besides loss to continue train
    ## ##############################################################
    output_dir = Path(args.output_dir)
    ckpt_path = output_dir / 'checkpoint.pth'
    if ckpt_path.exists():
        if args.no_resume:
            if is_main_process():
                print('*' * 10, f'Warning: overwriting {ckpt_path}', '*' * 10)
        else:
            if is_main_process():
                print(f'Resume from {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            start_epoch = checkpoint['epoch'] + 1
            global_step = start_epoch * len(train_dataloader)
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if is_main_process():
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(ckpt_path, checkpoint['epoch']))

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        for epoch in range(start_epoch, args.epochs):
            if epoch < args.train_epochs:
                train_sampler.set_epoch(epoch)
                tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                                   scheduler, global_step, local_rank=args.local_rank)
                if args.local_rank == 0:
                    logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                    R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
                    logger.info("the R1 is: {:.4f}".format(R1))

                    save_model(epoch, args, model, optimizer, tr_loss, type_name="", ckpt_path=ckpt_path,
                               scheduler=scheduler)
                    shutil.copy(ckpt_path, os.path.join(output_dir, 'ckpt_epoch{}.pth'.format(epoch + 1)))

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu)
    if is_main_process():
        run_obj.finish()


if __name__ == "__main__":
    args = get_args()
    if args.hf:
        mp.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
    else:
        main(args.local_rank, args)
