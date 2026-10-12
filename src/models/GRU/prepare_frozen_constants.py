"""Build/verify the shipped F1 full-training constants, CPU and no training."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def prepare(out):
    import eval_checkpoint as ec
    for key in list(ec.KEYS.values()) + ec.ENV_VARS_EXTRA:
        os.environ.pop(key, None)
    os.environ.update(VARIANTS='forward_direct', NUM_LAYERS='1', HIDDEN_SIZE='128', DROPOUT='0',
                      VAL_SPLIT='stratified', SPLIT_NAME='round10_vs3_s42.json', TAVG_MASK='split',
                      SEAL_TEST='1', TF32='0', TINNER_MODE='anchor', OTHER_CH_MODE='abs', SEEDS='7')
    for name in [name for name in list(sys.modules) if name.startswith('tes_gru')]:
        del sys.modules[name]
    from tes_gru import config
    from tes_gru.data import load_all_data, load_split_spec
    from tes_gru.models import build_model
    from tes_gru.train import fit_model_constants
    from tes_gru.runio import write_constants_sidecar, model_constants_dict
    from tes_gru.utils import _to_jsonable, sha256_lines, sha256_file
    train, val, test, scaler = load_all_data(include_test=False)
    if len(train) != 284 or len(val) != 20 or test:
        raise RuntimeError('full-training constants require sealed 284/20 protocol')
    model = build_model('forward_direct', config.LATEST_PARAMS, 'point').cpu()
    fit_model_constants(model, 'forward_direct', train, scaler)
    spec = load_split_spec()
    if os.path.isfile(out):
        with open(out, encoding='utf-8') as f:
            existing = json.load(f)
        expected = _to_jsonable(model_constants_dict(model))
        if existing['constants'] != expected:
            raise RuntimeError('existing immutable full-training constants differ from current fit')
        for name in ('scale_', 'min_', 'data_min_', 'data_max_'):
            if existing['scaler'][name] != getattr(scaler, name).tolist():
                raise RuntimeError('existing immutable full-training scaler differs: ' + name)
        print('VERIFIED immutable F1 constants: %s sha256=%s' % (out, sha256_file(out)))
        return out
    write_constants_sidecar(out, model, scaler, extra={'variant': 'forward_direct',
                           'frozen_constants_split': 'round10_vs3_s42.json',
                           'frozen_train_n': 284, 'frozen_train_sha256': sha256_lines(spec['train']),
                           'generated_without_training': True})
    print('PREPARED F1 constants: %s sha256=%s' % (out, sha256_file(out)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=os.path.join(HERE, 'round10_resources', 'full_train.constants.json'))
    args = parser.parse_args()
    prepare(os.path.abspath(args.out))


if __name__ == '__main__':
    main()
