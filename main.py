from argparse import ArgumentParser
from pytorch_lightning import Trainer
import torch
  
def main(hparams):
    model = torch.nn.Module()
    trainer = Trainer(gpus=hparams.gpus)
    trainer.fit(model)
  
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()
  
    main(args)



# cf. tmux사용법 : https://velog.io/@ur-luella/tmux-%EC%82%AC%EC%9A%A9%EB%B2%95
# cf. pytorch-lightning : https://baeseongsu.github.io/posts/pytorch-lightning-introduction/