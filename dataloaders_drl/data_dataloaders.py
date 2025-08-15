import torch
from torch.utils.data import DataLoader


def dataloader_msrvtt_train(args, tokenizer):
    if args.dataset_type == 'img':
        from .dataloader_msrvtt_retrieval_images import MSRVTTDataset
    else:
        from .dataloader_msrvtt_retrieval import MSRVTTDataset
    msrvtt_dataset = MSRVTTDataset(
        subset='train',
        # anno_path='data/anns',
        anno_path=args.anno_path,
        video_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=1,
        slice_framepos=args.slice_framepos,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    if args.dataset_type == 'img':
        from .dataloader_msrvtt_retrieval_images import MSRVTTDataset
    else:
        from .dataloader_msrvtt_retrieval import MSRVTTDataset
    msrvtt_testset = MSRVTTDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=1,
        slice_framepos=args.slice_framepos,
        config=args
    )

    test_sampler = None  # cpu
    dataloader = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader, len(msrvtt_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train": dataloader_msrvtt_train, "val": dataloader_msrvtt_test, "test": None}