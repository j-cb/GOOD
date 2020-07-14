def schedule_MNIST(args):
    schedule = dict([])
    schedule['epochs'] = 420
    schedule['optimizer'] = 'SGDM'
    schedule['start_lr'] = 0.005/args.batch_size
    schedule['lr_decay_epochs'] = [50, 100, 200, 300, 350]
    schedule['lr_decay_factor'] = 0.2
    schedule['kappa'] = 0.3
    schedule['out_start_epoch'] = 10000
    schedule['kappa_epoch_ramp'] = 0   
    schedule['eps'] = 0.3
    schedule['eps_start_epoch'] = 10000
    schedule['eps_epoch_ramp'] = 0
    if args.method == 'plain':
        pass
    if args.method in {'OE', 'CEDA', 'GOOD'}:
        schedule['out_start_epoch'] = 2
        schedule['kappa_epoch_ramp'] = 100
    if args.method in {'GOOD'}:
        schedule['eps_start_epoch'] = 10
        schedule['eps_epoch_ramp'] = 120
    if args.acet: #might need adjustment if not run with --method plain
        schedule['acet_n'] = 40
        schedule['out_start_epoch'] = 2
        schedule['kappa_epoch_ramp'] = 0
        schedule['eps_start_epoch'] = 6
        schedule['eps_epoch_ramp'] = 0
    return schedule

def schedule_CIFAR10(args):
    schedule = dict([])
    schedule['epochs'] = 420
    schedule['optimizer'] = 'ADAM'
    schedule['start_lr'] = 0.1/args.batch_size
    schedule['lr_decay_epochs'] = [30, 100]
    schedule['lr_decay_factor'] = 0.2
    schedule['kappa'] = 1.0
    schedule['out_start_epoch'] = 10000
    schedule['kappa_epoch_ramp'] = 0
    schedule['eps'] = 0.01
    schedule['eps_start_epoch'] = 10000
    schedule['eps_epoch_ramp'] = 0
    if args.method == 'plain':
        pass
    if args.method in {'OE', 'CEDA'}:
        schedule['out_start_epoch'] = 60
        schedule['kappa_epoch_ramp'] = 300
    if args.method in {'GOOD'}:
        assert args.pretrained
        schedule['epochs'] = 900
        schedule['start_lr'] = 1.28e-2/args.batch_size
        schedule['lr_decay_epochs'] = [450, 750, 850]
        schedule['out_start_epoch'] = -2
        schedule['kappa_epoch_ramp'] = 300
        schedule['eps_start_epoch'] = 4
        schedule['eps_epoch_ramp'] = 200
    if args.acet: #might need adjustment if not run with --method plain
        schedule['acet_n'] = 40
        schedule['out_start_epoch'] = 2
        schedule['kappa_epoch_ramp'] = 0
        schedule['eps_start_epoch'] = 6
        schedule['eps_epoch_ramp'] = 0
    return schedule
    
def default_schedule(args):
    if args.dset_in_name == 'MNIST':
        return schedule_MNIST(args)
    if args.dset_in_name == 'CIFAR10':
        return schedule_CIFAR10(args)
    if args.dset_in_name == 'SVHN':
        schedule = schedule_CIFAR10(args)
        if not args.pretrained:
            schedule['start_lr'] = 0.01/args.batch_size
        return schedule

def customized_schedule(schedule, args):
    for setting in ['start_lr', 'optimizer', 'epochs', 'kappa', 'eps']:
        set_value = getattr(args, setting)
        if set_value is not None:
            schedule[setting] = set_value
    return schedule

def schedule(args):
    return customized_schedule(default_schedule(args), args)