import argparse
import collections
import os
import re

import torch


def average_checkpoints(args):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      args: The args passed to the script.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(args.inputs)

    for fpath in args.inputs:
        print("Loading: ", fpath)
        state = torch.load(
            fpath,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state["model"]

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                if args.geometric_mean:
                    params_dict[k] *= p
                else:
                    params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            if args.geometric_mean:
                averaged_params[k].pow_(1/num_models)
            else:
                averaged_params[k].div_(num_models)
        else:
            if args.geometric_mean:
                averaged_params[k].pow_(1/num_models).round()
            else:
                averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def last_n_checkpoints(paths, n, update_based, upper_bound=None):
    assert len(paths) == 1
    path = paths[0]
    if update_based:
        pt_regexp = re.compile(r"checkpoint_\d+_(\d+)\.pt")
    else:
        pt_regexp = re.compile(r"checkpoint(\d+)\.pt")
    files = PathManager.ls(path)

    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            if upper_bound is None or sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))
    if len(entries) < n:
        raise Exception(
            "Found {} checkpoint files but need at least {}", len(entries), n
        )
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)[:n]]


def main():
    parser = argparse.ArgumentParser(
        description="Tool to average the params of input checkpoints to "
        "produce a new checkpoint",
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    parser.add_argument('--geometric_mean', action='store_true',
                        help='Should we do geometric mean instead of arithmetic mean?')
    args = parser.parse_args()
    print(args)

    new_state = average_checkpoints(args)
    torch.save(new_state, args.output)
    print("Finished writing averaged checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()