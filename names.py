def model_folder_name(expfolder, starttime, args, schedule):
    if args.method in {'OE', 'CEDA', 'GOOD'}:
        kappa_str = f'k{schedule["kappa"]}'
        out_str = f'o{args.dset_out_name}'
    else:
        kappa_str = ''
        out_str = ''
    if args.method == 'GOOD': 
        quantile_str = f'Q{int(100*args.good_quantile)}'
        eps_str = f'e{schedule["eps"]}'
    else:
        quantile_str = ''
        eps_str = ''
    if args.acet != '':
        acet_n = 40
        acet_str = 'ACET' + args.acet + str(acet_n)
        eps_str = f'e{schedule["eps"]}'
        kappa_str = f'k{schedule["kappa"]}'
        out_str = f'o{args.dset_out_name}'
    else:
        acet_str = ''
    if args.traincrop !=0:
        crop_str = f'c{args.traincrop}'
    else:
        crop_str = ''
    folder_name = f'{expfolder}/{starttime.strftime("%m-%d-%H-%M-%S")}M{args.method}{quantile_str}{kappa_str}{eps_str}A{args.arch}o{args.dset_out_name}a{args.autoaugment*"AA"}{crop_str}{acet_str}'
    return folder_name