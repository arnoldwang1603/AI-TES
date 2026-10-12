"""F11 array adapter, callable only inside paper_report's second access window.

The caller supplies every frozen test frame and prediction. This module has
no CLI or raw-data loader; model/physics parameters and training reference
statistics are metadata fixed before F9. All outputs are descriptive.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_common import (CHANNELS, COLS, bundle_arrays, convention_map,
                             save_plot, write_json, write_table)
from paper_report import assert_test_access

HERE = Path(__file__).resolve().parent


def _read_json(path):
    with open(path, encoding='utf-8-sig') as stream:
        return json.load(stream)


def _test_number(key):
    match = re.search(r'\((\d+)\)', str(key))
    if match is None:
        raise RuntimeError('F11 case has no frozen numerical ID: %s' % key)
    return int(match.group(1))


def _validate_arrays(bundles, frames):
    keys = set(frames)
    if len(keys) != 70 or {_test_number(k) for k in keys} != set(range(1, 71)):
        raise RuntimeError('F11 requires each of the frozen 70 case IDs exactly once')
    for key in keys:
        if not str(key).replace('\\', '/').startswith('Test_70_cases/'):
            raise RuntimeError('F11 frame is not a frozen test ID: %s' % key)
        frame = frames[key]
        values = frame[list(COLS) + ['Time (s)', 'Input Temperature (C)']].to_numpy(float)
        if len(values) < 2 or not np.isfinite(values).all() or np.any(np.diff(values[:, 3]) <= 0):
            raise RuntimeError('invalid preloaded F11 frame: %s' % key)
    seen, truth_by_arm, output = set(), {}, []
    for bundle in bundles:
        arm, seed = str(bundle['arm']), int(bundle['seed'])
        if (arm, seed) in seen:
            raise RuntimeError('duplicate F11 arm/seed: %s/%d' % (arm, seed))
        seen.add((arm, seed))
        ids, actuals, predictions = bundle_arrays(bundle)
        if len(ids) != 70 or len(set(ids)) != 70 or set(ids) != keys:
            raise RuntimeError('F11 bundle must contain the identical full 70-case ID set')
        mapping = {}
        for key, truth, prediction in zip(ids, actuals, predictions):
            target = frames[key][list(COLS)].to_numpy(float)[1:]
            if truth.shape != target.shape or prediction.shape != target.shape:
                raise RuntimeError('F11 prediction/target horizon or shape mismatch: %s' % key)
            if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
                raise RuntimeError('nonfinite F11 prediction/target: %s' % key)
            # Saved predictions use float32; this tolerance only checks their
            # raw-C round trip against the preloaded source, not model error.
            if not np.allclose(truth, target, atol=1e-4, rtol=0):
                raise RuntimeError('F11 stored truth disagrees with the preloaded frame: %s' % key)
            prior = truth_by_arm.setdefault((arm, key), truth)
            if not np.array_equal(prior, truth):
                raise RuntimeError('F11 truth changed across seeds: %s/%s' % (arm, key))
            mapping[key] = (truth, prediction)
        output.append(dict(bundle, arm=arm, seed=seed, arrays=mapping))
    if not output:
        raise RuntimeError('F11 has no prediction bundles')
    return output


def _recipe_config(bundle, records):
    record = records.get((bundle['arm'], bundle['seed']), {})
    config = bundle.get('config') or record.get('config')
    if config is None and record.get('run_dir'):
        config = _read_json(Path(record['run_dir']) / 'run_config.json')
    if config is None:
        raise RuntimeError('ensemble recipe cannot be verified without checkpoint configuration')
    return config


def _recipe_signature(config):
    knobs = ('variants', 'window_size', 'sliding_pad_mode', 'tinner_mode', 'touter_mode',
             'touter_tau', 'touter_scale', 'touter_ema_form', 'anchor_lead', 'loss_weights',
             'physics_bound_weight', 'other_ch_mode', 'input_lookahead', 'anchor_scale',
             'pos_gap_floor', 'pos_case_gate', 'pos_floor_soft', 'pos_temp_gate',
             'pos_temp_soft', 'pos_abs_head', 'pos_learned_gate', 'pos_gate_bias',
             'pos_learned_pool', 'pos_anchored_fallback', 'pos_fit_clean', 'case_flag_input',
             'exclude_both_phase', 'model_split', 'pos_time_prior', 'tf32', 'tavg_mask',
             'relabel_csv', 'loss_w0_mode', 'enc_widths', 'h0_mode', 'init_encoder',
             'feat_cols', 'early_step_weight', 'ratio_channels', 'ratio_ref', 'phys_mode',
             'init_burnin_steps', 'synthetic_rest_files', 'constants_from_sha256')
    model = {key: config.get(key) for key in knobs}
    model['latest_params'] = config.get('latest_params', {})
    data = config.get('data', {})
    model['data'] = {key: data.get(key) for key in ('val_split', 'val_per_type', 'split_name',
                     'split_sha256', 'train_exclude_type', 'frozen_train_sha256')}
    return json.dumps(model, sort_keys=True, separators=(',', ':'))


def _case_bootstrap(values, rng, draws=10000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError('case bootstrap requires finite case values')
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def _summary(per_file, model_column='model'):
    rng = np.random.default_rng(20261012)
    rows = []
    for (model, channel), part in per_file.groupby([model_column, 'channel'], sort=True):
        mean, lo, hi = _case_bootstrap(part['MAE'].to_numpy(float), rng)
        rows.append(dict(model=model, channel=channel, mean_MAE_C=mean,
                         case_bootstrap_lo_C=lo, case_bootstrap_hi_C=hi,
                         n_files=len(part), uncertainty='descriptive 10000 case bootstrap; fixed model/ensemble'))
    return pd.DataFrame(rows)


def _ensemble_outputs(groups, out):
    from error_budget import SWITCHED
    per_file, spreads = [], []
    rng = np.random.default_rng(20261012)
    summaries = []
    for name, members in groups:
        ordered_keys = sorted(members[0]['arrays'], key=_test_number)
        cases = {key: np.stack([b['arrays'][key][1] for b in members]) for key in ordered_keys}
        for key, predictions in cases.items():
            truth = members[0]['arrays'][key][0]
            if any(not np.array_equal(member['arrays'][key][0], truth) for member in members):
                raise RuntimeError('ensemble truth changed across frozen seed members: %s/%s' % (name, key))
            mean = predictions.mean(axis=0)
            std = predictions.std(axis=0, ddof=1)
            for index, channel in enumerate(CHANNELS):
                mae = float(np.mean(np.abs(mean[:, index] - truth[:, index])))
                single = float(np.mean(np.abs(predictions[:, :, index] - truth[:, index])))
                per_file.append(dict(ensemble=name, n_models=len(members), file=key,
                                     case_id=_test_number(key), channel=channel, MAE=mae,
                                     single_model_MAE_C=single, ensemble_gain_C=single-mae,
                                     mean_pointwise_seed_sd_C=float(std[:, index].mean()),
                                     max_pointwise_seed_sd_C=float(std[:, index].max())))
        subsets = {'all70': set(range(1, 71)), 'B6_registered4': {70, 42, 66, 67},
                   'B6_remaining66': set(range(1, 71)) - {70, 42, 66, 67},
                   'in_distribution66': set(range(1, 71)) - {54, 55, 56, 57},
                   'OOD4_54_57': {54, 55, 56, 57},
                   'both_phase21': set(SWITCHED), 'other49': set(range(1, 71)) - set(SWITCHED)}
        for subset, ids in subsets.items():
            selected = [key for key in ordered_keys if _test_number(key) in ids]
            for index, channel in enumerate(CHANNELS):
                per_seed = np.asarray([np.mean([np.mean(np.abs(b['arrays'][key][1][:, index]-b['arrays'][key][0][:, index]))
                                               for key in selected]) for b in members])
                ensemble_errors = np.asarray([np.mean(np.abs(cases[key].mean(axis=0)[:, index]-members[0]['arrays'][key][0][:, index]))
                                              for key in selected])
                mean, lo, hi = _case_bootstrap(ensemble_errors, rng)
                summaries.append(dict(ensemble=name, subset=subset, channel=channel,
                                      n_models=len(members), n_files=len(selected), MAE=mean,
                                      case_bootstrap_lo_C=lo, case_bootstrap_hi_C=hi,
                                      seed_MAE_mean_C=float(per_seed.mean()), seed_MAE_sd_C=float(per_seed.std(ddof=1)),
                                      seed_MAE_min_C=float(per_seed.min()), seed_MAE_max_C=float(per_seed.max()),
                                      uncertainty='fixed seed membership; descriptive case uncertainty and seed spread'))
                for member, value in zip(members, per_seed):
                    spreads.append(dict(ensemble=name, subset=subset, channel=channel,
                                        arm=member['arm'], seed=member['seed'], n_files=len(selected), MAE=float(value)))
        # Retain complete ensemble predictions for checking and replotting.
        preds = [cases[key].mean(axis=0) for key in ordered_keys]
        actuals = [members[0]['arrays'][key][0] for key in ordered_keys]
        def stored_arrays(arrays):
            if len({a.shape for a in arrays}) == 1:
                return np.stack(arrays)
            result = np.empty(len(arrays), dtype=object)
            result[:] = arrays
            return result
        np.savez_compressed(out / (name + '_predictions.npz'),
                            case_ids=np.asarray(ordered_keys, dtype=object),
                            actuals=stored_arrays(actuals), preds=stored_arrays(preds),
                            member_arms=np.asarray([m['arm'] for m in members]),
                            member_seeds=np.asarray([m['seed'] for m in members]))
    return pd.DataFrame(per_file), pd.DataFrame(summaries), pd.DataFrame(spreads)


def finish_f11(manifest, bundles, out, frames):
    """Complete the registered final report using already-loaded test arrays."""
    lease = assert_test_access()
    if lease['mode'] != 'F11':
        raise RuntimeError('F11 array analysis requires the registered F11 capability')
    checked = _validate_arrays(bundles, frames)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = manifest.get('prereg')
    if not isinstance(protocol, dict):
        protocol = _read_json(protocol or HERE / 'prereg/round10_v3.json')
    recipe = manifest.get('frozen_recipe')
    if not isinstance(recipe, dict) or not recipe.get('arm'):
        raise RuntimeError('F11 requires the recipe frozen before the fresh seeds')
    source = recipe['arm']
    direct = source in ('direct', 'B1_mask', 'B2_mask')
    source_arm = 'B1_mask' if direct else 'C17_' + source
    fresh = set(map(int, protocol['seeds']['F9_fresh_primes']))
    old = set(map(int, protocol['seeds']['S10']))
    if len(fresh) != 20 or len(old) != 10 or fresh & old:
        raise RuntimeError('F11 requires the registered disjoint 20 fresh and 10 existing seeds')
    by_arm = {}
    for bundle in checked:
        by_arm.setdefault(bundle['arm'], []).append(bundle)
    def members(arm, seeds):
        result = by_arm.get(arm, [])
        if {b['seed'] for b in result} != seeds or len(result) != len(seeds):
            raise RuntimeError('F11 missing/extra registered seed in ensemble arm ' + arm)
        return sorted(result, key=lambda b: b['seed'])
    f9 = members('F9', fresh)
    existing = members(source_arm, old)
    members('B1_mask', old)
    records = {(str(r['arm']), int(r['seed'])): r for r in manifest.get('runs', [])}
    groups = [('F9_20', f9), ('recipe_30', existing + f9)]
    if not direct:
        f10 = members('F10', fresh)
        groups += [('F10_direct_20', f10), ('direct_30', members('B1_mask', old) + f10)]
    for name, group in groups:
        signatures = {_recipe_signature(_recipe_config(bundle, records)) for bundle in group}
        if len(signatures) != 1:
            raise RuntimeError('mixed model recipes in frozen ensemble ' + name)
    analyses = Path(manifest.get('analyses_root') or '')
    if not manifest.get('analyses_root'):
        raise RuntimeError('F11 needs the frozen S5b analysis resources')
    conventions = convention_map(manifest.get('conventions'))
    metadata_keys = set(frames) - set(conventions)
    if metadata_keys:
        raise RuntimeError('F11 convention metadata missing %d files' % len(metadata_keys))
    strict_keys = sorted(key for key in frames if conventions[key].get('label') == 'line average')
    if len(strict_keys) != 67:
        raise RuntimeError('F11 strict line-average yardstick must contain 67 files')
    from physics_reference import from_frame
    params = _read_json(manifest.get('physics_params') or HERE / 'round10_resources/physics/params.json')
    parameters = {key: params[key] for key in ('r_o', 'n_layers', 'sub_steps') if key in params}
    physics = {key: from_frame(frame, k_scale=float(params.get('s', 1)), params=parameters)
               for key, frame in sorted(frames.items())}
    physical_rows = []
    for key, frame in frames.items():
        truth = frame[list(COLS)].to_numpy(float)[1:]
        p = np.column_stack([physics[key][name][1:] for name in ('T_inner_phys', 'T_outer_phys', 'T_avg_line_phys')])
        if p.shape != truth.shape or not np.isfinite(p).all():
            raise RuntimeError('invalid E1 physics horizon: ' + key)
        for index, channel in enumerate(CHANNELS):
            error = p[:, index] - truth[:, index]
            physical_rows.append(dict(model='E1_physics', file=key, channel=channel,
                                      MAE=float(np.mean(abs(error))), RMSE=float(np.sqrt(np.mean(error**2))),
                                      MaxErr=float(abs(error).max())))
    physical_table = pd.DataFrame(physical_rows)
    from linear_sysid import load_models, score_frames
    models = load_models(analyses / 'linear_sysid/linear_sysid_models.json')
    if set(models) != {'linear_global', 'linear_binned'}:
        raise RuntimeError('F11 requires both validation-selected linear baseline models')
    linear = []
    for name in ('linear_global', 'linear_binned'):
        values = score_frames(models[name], frames)
        values['model'] = name
        if len(values) != 210:
            raise RuntimeError('linear F11 rows must cover all70 cases and3 channels')
        linear.append(values)
    linear_table = pd.concat(linear, ignore_index=True)
    baseline_summary = _summary(pd.concat([physical_table, linear_table], ignore_index=True))
    from c16_curves import test_points
    reference = pd.read_csv(analyses / 'c16_curves/c16_reference_statistics.csv')
    confirmed = set(manifest.get('confirmed', []))
    if not direct and source not in confirmed:
        raise RuntimeError('frozen challenger recipe was not confirmed before F9')
    aliases = {'F9': 'B2_mask' if direct else source, 'F10': 'B2_mask'}
    for arm in confirmed:
        aliases['C17_' + arm] = arm
    selected, references = [], []
    for alias, original in sorted(aliases.items()):
        part = by_arm.get(alias, [])
        if not part:
            if alias.startswith('C17_'):
                raise RuntimeError('confirmed family has no F11 checkpoint: ' + alias)
            continue
        source_rows = reference[reference['arm'] == original].copy()
        if set(source_rows['channel']) != set(CHANNELS) or len(source_rows) != 3:
            raise RuntimeError('missing frozen training reference statistics for ' + original)
        source_rows['reference_arm'] = original
        source_rows['arm'] = alias
        references.append(source_rows)
        selected.extend(part)
    c16 = test_points(selected, pd.concat(references, ignore_index=True))
    if not np.isfinite(c16['reference_residual_std_C'].to_numpy(float)).all():
        raise RuntimeError('C16 test points must use complete frozen training reference statistics')
    from error_budget import analyze_bundles, SWITCHED
    budget = analyze_bundles(by_arm['B1_mask'] + by_arm['F9'], frames, physics,
                             conventions=conventions, which_set='test')
    floors = budget['floors']
    avg_floor = floors[floors['channel'] == 'T_avg'].copy()
    avg_floor['residual_upper_bound_RMSE_C'] = avg_floor['label_upper_bound_RMSE_C']
    avg_floor['upper_bound_source'] = 'frozen convention table tavg_rmse_best_mix_C after per-file inner-share fit'
    avg_floor['upper_bound_interpretation'] = 'registered per-file residual upper bound after fitting the label averaging share; not an independently measured repeat-noise floor'
    avg_floor['physics_yardstick_interpretation'] = 'raw 1-D line-average discrepancy before averaging-share fitting; reported separately from the registered residual upper bound'
    strict = avg_floor[avg_floor['strict_line_average']].copy()
    oracle21 = budget['oracle'][budget['oracle']['case_id'].isin(SWITCHED)].copy()
    if len(strict) != 67 or len(oracle21) != 21:
        raise RuntimeError('F11 yardstick/oracle membership differs from registered67/21')
    ensemble_per_file, ensemble_summary, seed_spread = _ensemble_outputs(groups, out)
    tables = {'E1_physics_per_file': physical_table, 'linear_test_per_file': linear_table,
              'baseline_test_summary': baseline_summary, 'C16_test_points': c16,
              'error_budget': budget['budget'], 'error_floors': floors,
              'strict67_line_average_yardstick': strict, 'bracket21_oracle': oracle21,
              'per_file_residual_upper_bound': avg_floor, 'ensemble_per_file': ensemble_per_file,
              'ensemble_summary': ensemble_summary, 'ensemble_subset_seed_spread': seed_spread}
    for name, frame in tables.items():
        note = 'F11 preloaded arrays; descriptive readout after recipe freeze. No fresh-seed selection.'
        if name == 'C16_test_points':
            from round10_refit import REFIT_SCOPE
            note += ' ' + REFIT_SCOPE
        write_table(frame, out, name, note)
    def baseline_plot(plt):
        figure, axis = plt.subplots(figsize=(8, 4))
        baseline_summary.pivot(index='channel', columns='model', values='mean_MAE_C').plot.bar(ax=axis)
        axis.set_ylabel('Test MAE (C)'); axis.tick_params(axis='x', rotation=0)
        return figure
    save_plot(out / 'F11_baselines.png', baseline_plot)
    def reference_plot(plt):
        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        for axis, channel in zip(axes, CHANNELS):
            part = c16[c16['channel'] == channel]
            axis.scatter(part['reference_residual_std_C'], part['model_MAE_C'])
            for _, row in part.iterrows():
                axis.annotate(row['arm'], (row['reference_residual_std_C'], row['model_MAE_C']), fontsize=7)
            axis.set_title(channel); axis.set_xlabel('Frozen training reference std (C)'); axis.set_ylabel('Test MAE (C)')
        return figure
    save_plot(out / 'F11_C16_reference_quality.png', reference_plot)
    def ensemble_plot(plt):
        figure, axis = plt.subplots(figsize=(8, 4))
        part = ensemble_summary[ensemble_summary['subset'] == 'all70']
        part.pivot(index='channel', columns='ensemble', values='MAE').plot.bar(ax=axis)
        axis.set_ylabel('Fixed ensemble test MAE (C)'); axis.tick_params(axis='x', rotation=0)
        return figure
    save_plot(out / 'F11_ensembles.png', ensemble_plot)
    membership = [{'ensemble': name, 'members': [{'arm': b['arm'], 'seed': b['seed']} for b in group]}
                  for name, group in groups]
    write_json(out / 'ensemble_membership.json', membership)
    result = dict(artifacts={name: str(out / (name + '.csv')) for name in tables},
                  plots=[str(out / name) for name in ('F11_baselines.png', 'F11_C16_reference_quality.png', 'F11_ensembles.png')],
                  yardsticks=dict(strict_line_average_n=67, strict_case_ids=sorted(_test_number(k) for k in strict_keys),
                                  bracket_oracle_n=21, bracket_case_ids=sorted(SWITCHED)),
                  ensemble_metrics=ensemble_summary[ensemble_summary['subset'] == 'all70'].to_dict('records'),
                  membership=str(out / 'ensemble_membership.json'),
                  truth_roundtrip_tolerance_C=1e-4, truth_across_seeds='bit-identical within each arm')
    write_json(out / 'F11_analyses.json', result)
    return result
