import torchvision
import torchvision.transforms as transforms

#Loading the dataset and preprocessing
torchvision.datasets.MNIST(root = './data',train = True,transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),download = True)


torchvision.datasets.MNIST(root = './data',train = False,transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),)

