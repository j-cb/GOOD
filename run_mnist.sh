sleep 3s; nohup python3.7 goodX.py --gpu 0 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'plain' > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 0 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'CEDA' > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 0 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'OE' > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 1 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'plain' --acet 'ce' > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 1 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'GOOD' --good_quantile 0.0 > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 1 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'GOOD' --good_quantile 0.2 > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 2 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'GOOD' --good_quantile 0.4 > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 2 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'GOOD' --good_quantile 0.6 > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 2 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'GOOD' --good_quantile 0.8 > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 3 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'GOOD' --good_quantile 0.9 > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 3 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'GOOD' --good_quantile 0.95 > logs/$(date +%Y-%m-%d_%H:%M:%S).log &
sleep 3s; nohup python3.7 goodX.py --gpu 3 --dset_in_name 'MNIST' --arch 'L' --mode 'train' --method 'GOOD' --good_quantile 1.0 > logs/$(date +%Y-%m-%d_%H:%M:%S).log &