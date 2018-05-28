import torchvision.models as models
import torch.utils.model_zoo as model_zoo


load = model_zoo.load_url


def load_url(url, model_dir=None, map_location=None):
    url = url.replace('https', 'http')
    print url
    return load(url)

model_names = sorted(name for name in models.__dict__
                     if not (not (name.islower() and not name.startswith("__"))
                             or not callable(models.__dict__[name])))


def main():
    model_zoo.load_url = load_url
    models.alexnet(True)
    for model_fun in model_names:
        models.__dict__[model_fun](True)


if __name__ == '__main__':
    main()
