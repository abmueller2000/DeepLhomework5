import pystk
import numpy as np
from itertools import product
from homework.utils import PyTux

def control(aim_point, current_vel, params):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    # Instantiate the action object
    action = pystk.Action()
    target_velocity = params['target_velocity']
    steer_scale = params['steer_scale']
    max_steering_angle = params['max_steering_angle']
    drift_threshold = params['drift_threshold']
    nitro_threshold = params['nitro_threshold']

    # # Constants
    # target_velocity = 25
    # steer_scale = 2
    # max_steering_angle = 1
    # drift_threshold = 0.8  # threshold for drifting off course
    # nitro_threshold = 0.1  # threshold for using nitro when going almost straight

    # Calculate the difference between the current velocity and the target velocity
    velocity_diff = target_velocity - current_vel

    # Set acceleration to 1 if below target velocity, else 0
    action.acceleration = 1.0 if velocity_diff > 0 else 0

    # Calculate steering angle based on the aim point, scale it by steer_scale
    # and make sure it's within the allowed range of -1 to 1
    steer_angle = aim_point[0] * steer_scale
    steer_angle = max(min(steer_angle, max_steering_angle), -max_steering_angle)
    action.steer = steer_angle

    # Determine if we should drift and brake only if drifting significantly off course
    action.drift = abs(steer_angle) > drift_threshold
    action.brake = action.drift and current_vel > target_velocity

    # Use nitro if going straight or almost straight
    action.nitro = abs(steer_angle) < nitro_threshold

    return action


def optimize_parameters(pytux, track, max_frames=1000, verbose=False):
    """
    Optimize the parameters for the control function on a given track.
    :param pytux: An existing PyTux instance to use for the simulation
    :param track: The track name to optimize parameters for
    :param max_frames: The maximum number of frames to simulate
    :param verbose: If True, print detailed output
    :return: Tuple of best parameters and the corresponding time
    """
    parameter_ranges = {
        'target_velocity': np.arange(20, 30, 1),
        'steer_scale': np.arange(1, 3, 0.5),
        'max_steering_angle': np.arange(0.5, 2, 0.5),
        'drift_threshold': np.arange(0.5, 1, 0.1),
        'nitro_threshold': np.arange(0, 0.2, 0.05)
    }

    best_time = float('inf')
    best_params = None

    # Grid search over the parameter space
    for params in product(*parameter_ranges.values()):
        param_dict = dict(zip(parameter_ranges.keys(), params))
        def param_control(aim_point, current_vel):
            return control(aim_point, current_vel, param_dict)

        steps, distance = pytux.rollout(track, controller=param_control, max_frames=max_frames, verbose=verbose)
        time = steps  # Assuming steps is a proxy for time

        if verbose:
            print(f'Tested params: {param_dict}, Steps: {steps}, Distance: {distance}')

        if time < best_time:
            best_time = time
            best_params = param_dict

    pytux.close()
    return best_params, best_time

if __name__ == '__main__':
    track_to_optimize = 'lighthouse'  # Replace with any track you want to optimize for
    pytux = PyTux()
    best_params, best_time = optimize_parameters(pytux, track=track_to_optimize, max_frames=1000, verbose=True)
    print(f'Best Parameters for {track_to_optimize}: {best_params}')
    print(f'Best Time for {track_to_optimize}: {best_time}')
    pytux.close()  # Close the PyTux instance
