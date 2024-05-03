from .mydataset import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def build_dataset(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )
    # print(flag, len(data_set))
    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     drop_last=drop_last)
    return data_set, shuffle_flag