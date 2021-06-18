import torch



def count_flops_params(model, x, custom_ops=None, verbose=True, mode='default'):
    """
    Count FLOPs and Params of the given model. This function would
    identify the mask on the module and take the pruned shape into consideration.
    Note that, for sturctured pruning, we only identify the remained filters
    according to its mask, and do not take the pruned input channels into consideration,
    so the calculated FLOPs  will be larger than real number.
    Parameters
    ---------
    model : nn.Module
        Target model.
    x : tuple or tensor
        The input shape of data (a tuple), a tensor or a tuple of tensor as input data.
    custom_ops : dict
        A mapping of (module -> torch.nn.Module : custom operation)
        the custom operation is a callback funtion to calculate
        the module flops and parameters, it will overwrite the default operation.
        for reference, please see ``ops`` in ``ModelProfiler``.
    verbose : bool
        If False, mute detail information about modules. Default is True.
    mode : str
        the mode of how to collect information. If the mode is set to ``default``,
        only the information of convolution and linear will be collected.
        If the mode is set to ``full``, other operations will also be collected.
    Returns
    -------
    tuple of int, int and dict
        Representing total FLOPs, total parameters, and a detailed list of results respectively.
        The list of results are a list of dict, each of which contains (name, module_type, weight_shape,
        flops, params, input_size, output_size) as its keys.
    """

    assert isinstance(x, tuple) or isinstance(x, torch.Tensor)
    assert mode in ['default', 'full']

    original_device = next(model.parameters()).device
    training = model.training

    if isinstance(x, tuple) and all(isinstance(t, int) for t in x):
        x = (torch.zeros(x).to(original_device), )
    elif torch.is_tensor(x):
        x = (x.to(original_device), )
    else:
        x = (t.to(original_device) for t in x)

    handler_collection = []
    profiler = ModelProfiler(custom_ops, mode)

    prev_m = None
    for name, m in model.named_modules():
        # dealing with weight mask here
        if isinstance(prev_m, PrunerModuleWrapper):
            # weight mask is set to weight mask of its parent (wrapper)
            weight_mask = prev_m.weight_mask
            m.weight_mask = weight_mask
        prev_m = m

        if type(m) in profiler.ops:
            # if a leaf node
            _handler = m.register_forward_hook(functools.partial(profiler.count_module, name=name))
            handler_collection.append(_handler)

    model.eval()

    with torch.no_grad():
        model(*x)

    # restore origin status
    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    if verbose:
        # get detail information
        print(profiler.format_results())
        print(f'FLOPs total: {profiler.sum_flops()}')
        print(f'#Params total: {profiler.sum_params()}')

    return profiler.sum_flops(), profiler.sum_params(), profiler.results