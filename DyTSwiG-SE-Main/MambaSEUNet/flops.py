import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
from models.generator import *
from thop import profile
from utils.util import load_config

torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='MambaSEUNet_emb_32')
    parser.add_argument('--config', default='/home/xyj/Experience/PrimeK-Net-main/config_SEMUet.json')
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device('cuda:{:d}'.format(0))

    model = MambaSEUNet(cfg).to(device)

    num_params = sum(p.numel() for p in model.parameters())

    print(f"Manual calculation of parameters: {num_params}")

    with torch.no_grad():
        dummy_input1 = torch.rand(1, 256, 256).to(device)
        dummy_input2 = torch.rand(1, 256, 256).to(device)
        flops, params = profile(model, inputs=(dummy_input1, dummy_input2))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

if __name__ == '__main__':
    main()
