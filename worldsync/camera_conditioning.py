"""Fixed-scale native camera conditioning for a declared canonical episode plan."""
import math


def plan_translation_scale(timeline, move_speed=1.0):
    """Choose once before rendering; corrections rotate and cannot change this scale.

    Extensions repeat their original phase's displacement. Early release may mean
    the maximum is never reached; retaining the declared scale is intentional.
    """
    if not math.isfinite(move_speed) or move_speed <= 0:
        raise ValueError('Movement speed must be finite and positive')
    magnitudes = [move_speed * p['source']['duration_ms'] / (250 * p['steps'])
                  for p in timeline.phases if p['action'] in {'W', 'A', 'S', 'D'}]
    return max(magnitudes, default=1.0)


def fixed_scale_relative_poses(native_relative, poses, translation_scale):
    """Use the native transform, with one episode-wide translation denominator.

    Do not recompute the denominator from each prefix: that changes earlier
    movement units when a later phase uses a different fractional interval.
    """
    if not math.isfinite(translation_scale) or translation_scale <= 0:
        raise ValueError('Translation scale must be finite and positive')
    relative = native_relative(poses, framewise=True, normalize_trans=False)
    relative[:, :3, 3] /= translation_scale
    return relative
