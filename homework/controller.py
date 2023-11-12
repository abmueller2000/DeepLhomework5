import pystk
import numpy as np
from itertools import product
from homework.utils import PyTux

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift, nitro)
    """
    # Instantiate the action object
    action = pystk.Action()

    # Constants
    target_velocity = 36
    steer_scale = 3
    max_steering_angle = 1
    drift_threshold = 0.70  # Threshold for drifting off course
    nitro_threshold = 0.20  # Threshold for using nitro when going almost straight

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
    # action.brake = action.drift and current_vel > target_velocity

    # Use nitro if going straight or almost straight
    action.nitro = abs(steer_angle) < nitro_threshold

    return action

if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(f'Track: {t}, Steps: {steps}, Distance: {how_far}')
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
