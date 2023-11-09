from homework.controller import control, optimize_parameters
from homework.utils import PyTux
import numpy as np

def rollout_all_tracks(pytux, max_frames=1000, verbose=False):
    reqs = {
        'lighthouse': 550,
        'hacienda': 700,
        'snowtuxpeak': 700,
        'zengarden': 600,
        'cornfield_crossing': 750,
        'scotland': 700
    }

    results = {}
    for track in reqs.keys():
        print(f"Optimizing parameters for track: {track}")
        best_params, best_time = optimize_parameters(pytux, track, max_frames=max_frames, verbose=verbose)
        print(f"Best parameters for track {track}: {best_params} with time {best_time}")

        # Use the optimized parameters for the controller
        def controller(aim_point, current_vel):
            return control(aim_point, current_vel, best_params)

        frames, dist = pytux.rollout(track, controller=controller, max_frames=max_frames, verbose=verbose)
        results[track] = {'frames': frames, 'distance': dist, 'params': best_params}

    return results

if __name__ == '__main__':
    # Run the rollout for all tracks
    pytux = PyTux()
    all_track_results = rollout_all_tracks(pytux=pytux, max_frames=1000, verbose=True)
    pytux.close()

    # Print the results
    for track, result in all_track_results.items():
        print(f"Track: {track}, Frames: {result['frames']}, Distance: {result['distance']}, Params: {result['params']}")
