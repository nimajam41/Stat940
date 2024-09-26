import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
from tqdm.auto import tqdm, trange
import plotly.graph_objects as go
import plotly.express as px


def pointnet_loss(pred, labels, rot64, device, alpha = 0.001):
    criterion = nn.CrossEntropyLoss()
    batch_size = pred.shape[0]

    i64 = torch.eye(64, requires_grad=True, device=device).repeat(batch_size, 1, 1)
    mat64 = torch.bmm(rot64, rot64.transpose(1, 2))
    dif64 = nn.MSELoss(reduction='sum')(mat64, i64) / batch_size

    loss1 = criterion(pred, labels)
    loss2 = dif64
    loss = loss1 + alpha * loss2

    return loss


def dgcnn_loss(pred, gold, smoothing=True):
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1. - eps) + (1. - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()

    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


LOG_INTERVAL = 20


def train(model, optimizer, device, trainloader, model_name, verbose=True):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(tqdm(trainloader, desc='Batches', leave=False)):
        points, labels = data
        points = points.to(device)
        labels = labels.to(device)
        points = points.transpose(1, 2).float()
        optimizer.zero_grad()

        if model_name == "pointnet":
            logits, rot3, rot64 = model(points)
            loss = pointnet_loss(logits, labels, rot64, device)

        elif model_name == "dgcnn":
            logits = model(points)
            loss = dgcnn_loss(logits, labels, False)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if verbose and batch_idx % LOG_INTERVAL == LOG_INTERVAL - 1:
            print('    Train [%d/%d]\t | \tLoss: %.5f' % (
            batch_idx * logits.shape[0], len(trainloader.dataset), loss.item()))

    train_loss /= batch_idx

    if verbose:
        print('==> Train | Average loss: %.4f' % train_loss)

    return train_loss


def test(model, testloader, device, model_name, verbose=True):
    model.eval()
    test_loss = 0

    total = 0
    correct = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            points, labels = data
            points = points.to(device)
            labels = labels.to(device)
            points = points.transpose(1, 2).float()

            if model_name == "pointnet":
                logits, rot3, rot64 = model(points)
                loss = pointnet_loss(logits, labels, rot64, device)

            elif model_name == "dgcnn":
                logits = model(points)
                loss = dgcnn_loss(logits, labels, False)

            _, predicted = torch.max(logits.data, 1)
            total += labels.shape[0]
            correct += (labels == predicted).sum().item()
            test_loss += loss.item()

        test_loss /= i
        acc = 100 * (correct / total)

        if verbose:
            print('==> Test  | Average loss: %.4f' % test_loss)
            print('==> Test  | Accuracy: %.4f' % acc)

        return test_loss, acc
        
        
def set_seed(seed=1):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(model, n_epoch, device, trainloader, testloader, model_name, checkpoint_path, optimizer_state_dict=None, verbose=True):
    model.to(device)
    lr = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    train_hist = []
    test_hist = []
    best_acc = 0
    best_epoch = 0

    for epoch in trange(0, n_epoch + 1, desc='Epochs', leave=True):
        if epoch % 20 == 19:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if verbose:
            print('\nEpoch %d:' % epoch)

        train_loss = train(model, optimizer, device, trainloader, model_name, verbose)
        test_loss, acc = test(model, testloader, device, model_name, verbose)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path + '/checkpoint.pt')

        if acc >= best_acc:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path + '/best.pt')
            best_acc = acc
            best_epoch = epoch


        train_hist.append(train_loss)
        test_hist.append(test_loss)

        print(f"Best epoch: {best_epoch}")
        print(f"Best accuracy: {best_acc}")

    return train_hist, test_hist 


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig


def pcshow(xs,ys,zs, color="DarkSlateGray"):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      color=color),
                      selector=dict(mode='markers'))
    fig.show()