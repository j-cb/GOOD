import re

eval_paths = dict([]) #these should be lists of path strings of .pt files
eval_paths['MNIST'] = [
    'experiments/GOOD_MNIST/10-14-18-34-09MplainALoTINYac4/state_dicts/419fin.pt',
    'experiments/GOOD_MNIST/10-14-18-34-12MCEDAk0.3ALoTINYac4/state_dicts/419fin.pt',
    'experiments/GOOD_MNIST/10-14-18-34-15MOEk0.3ALoTINYac4/state_dicts/419fin.pt',
    'experiments/GOOD_MNIST/10-14-18-34-19Mplaink0.3e0.3ALoTINYac4ACETce40/state_dicts/200.pt',
    'experiments/GOOD_MNIST/10-14-18-34-22MGOODQ0k0.3e0.3ALoTINYac4/state_dicts/419fin.pt',
    'experiments/GOOD_MNIST/10-14-18-34-42MGOODQ20k0.3e0.3ALoTINYac4/state_dicts/419fin.pt',
    'experiments/GOOD_MNIST/10-14-18-34-42MGOODQ60k0.3e0.3ALoTINYac4/state_dicts/419fin.pt',
    'experiments/GOOD_MNIST/10-14-18-34-43MGOODQ40k0.3e0.3ALoTINYac4/state_dicts/419fin.pt',
    'experiments/GOOD_MNIST/10-14-18-34-43MGOODQ80k0.3e0.3ALoTINYac4/state_dicts/419fin.pt',
    'experiments/GOOD_MNIST/10-14-18-34-43MGOODQ90k0.3e0.3ALoTINYac4/state_dicts/419fin.pt',
    'experiments/GOOD_MNIST/10-14-18-34-44MGOODQ95k0.3e0.3ALoTINYac4/state_dicts/419fin.pt',
    'experiments/GOOD_MNIST/10-14-18-34-48MGOODQ100k0.3e0.3ALoTINYac4/state_dicts/419fin.pt',
]

eval_paths['SVHN'] = [
    'experiments/GOOD_SVHN/05-26-13-23-42plainLlogit_spreade0.05k0.1bs128AXwd0.005oTinyImagesmN/state_dicts/419fin.pt',
    'experiments/GOOD_SVHN/10-14-10-34-04MCEDAk1.0AXLoTINYaAAc4/state_dicts/419fin.pt',
    'experiments/GOOD_SVHN/10-10-22-50-07MOEk1.0AXLoTINYaAAc4/state_dicts/419fin.pt',
    'experiments/GOOD_SVHN/10-10-22-50-06Mplaink1.0e0.03AXLoTINYaAAc4ACETce40/state_dicts/200.pt',
    'experiments/GOOD_SVHN/10-13-17-40-42MGOODQ0k1.0e0.03AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_SVHN/10-10-22-48-49MGOODQ20k1.0e0.03AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_SVHN/10-10-22-47-18MGOODQ40k1.0e0.03AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_SVHN/10-10-22-47-18MGOODQ60k1.0e0.03AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_SVHN/10-10-22-47-17MGOODQ80k1.0e0.03AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_SVHN/10-10-22-47-18MGOODQ90k1.0e0.03AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_SVHN/10-10-22-47-17MGOODQ95k1.0e0.03AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_SVHN/10-10-22-47-20MGOODQ100k1.0e0.03AXLoTINYaAAc4/state_dicts/899fin.pt',
]

eval_paths['CIFAR10'] = [
    '/mnt/SHARED/Julian/models_for_eval/provable_lc_CIFAR10/05-25-18-47-14plainLlogit_spreade0.05k0.1bs128AXwd0.005oTinyImagesmN/state_dicts/419fin.pt',
    'experiments/GOOD_CIFAR10/10-08-16-31-33MCEDAk1.0AXLoTINYaAAc4/state_dicts/419fin.pt',
    'experiments/GOOD_CIFAR10/10-08-16-31-33MOEk1.0AXLoTINYaAAc4/state_dicts/419fin.pt',
    'experiments/GOOD_CIFAR10/10-08-16-31-35Mplaink1.0e0.01AXLoTINYaAAc4ACETce40/state_dicts/200.pt',
    'experiments/GOOD_CIFAR10/10-13-18-15-33MGOODQ0k1.0e0.01AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_CIFAR10/10-08-16-31-46MGOODQ20k1.0e0.01AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_CIFAR10/10-08-16-31-49MGOODQ40k1.0e0.01AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_CIFAR10/10-08-16-31-52MGOODQ60k1.0e0.01AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_CIFAR10/10-08-16-31-56MGOODQ80k1.0e0.01AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_CIFAR10/10-08-16-33-18MGOODQ90k1.0e0.01AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_CIFAR10/10-08-16-33-19MGOODQ95k1.0e0.01AXLoTINYaAAc4/state_dicts/899fin.pt',
    'experiments/GOOD_CIFAR10/10-08-16-33-20MGOODQ100k1.0e0.01AXLoTINYaAAc4/state_dicts/899fin.pt',
]


def get_arch(path):
    if 'AL' in path:
        assert'AX' not in path
        return 'L'
    elif 'AX' in path:
        return 'XL'
    else:
        raise ValueError('No valid arch substring in the path.')
        
def get_shortname(s):
    match = re.search(r'\d\d-\d\d-\d\d-\d\d-\d\d\D(.*)[ek]\d', s)
    return match.group(1)

def get_shortname(s):
    if 'GOODQ' in s:
        match = re.search(r'GOODQ\d+', s)
        name = match.group(0)
    elif 'CEDA' in s:
        name = 'CEDA'
    elif 'OE' in s:
        name = 'OE'
    elif 'plain' in s:
        name = 'Plain'
    else:
        return 'Unknown'
    if 'ACET' in s:
        if name == 'Plain':
            name = 'ACET' #+  s.split('acetce')[1].split('/')[0]
        else:
            name += 'ACET'
    return name