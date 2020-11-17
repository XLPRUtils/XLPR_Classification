from torch import optim

def get_optimizer(type, model, lr, wd):
    if type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    elif type == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=wd)
    elif type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)
    elif type == 'Rprop':
        optimizer = optim.Rprop(model.parameters(), lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    elif type == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=wd, initial_accumulator_value=0)
    elif type == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=wd)
    elif type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=wd, momentum=0, centered=False)
    elif typpe == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
    elif type == 'SparseAdam':
        optimizer = torch.optim.SparseAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08)
    elif type == 'LBFGS' :
        optimizer = optim.LBFGS(params, lr=lr, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09,
                    history_size=100, line_search_fn=None)

    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)

    return optimizer