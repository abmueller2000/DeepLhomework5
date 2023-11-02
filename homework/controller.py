import pystk


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    # Constants
    target_velocity = 20  # You may need to tune this for optimal performance
    steer_scale = 1.0     # You may need to tune this scale factor
    max_steering_angle = 1.0  # Maximum steering angle

    # Calculate the difference between the current velocity and the target velocity
    velocity_diff = target_velocity - current_vel

    # Accelerate or brake based on the velocity difference
    if velocity_diff > 0:
        action.acceleration = min(velocity_diff, 1.0)
        action.brake = False
    else:
        action.acceleration = 0
        action.brake = True

    # Calculate steering angle based on the aim point, scale it by steer_scale
    # and make sure it's within the allowed range of -1 to 1
    steer_angle = aim_point[0] * steer_scale
    action.steer = max(min(steer_angle, max_steering_angle), -max_steering_angle)

    # Determine if we should drift
    action.drift = abs(steer_angle) > 0.5  # Drift if steering angle is large, tune this threshold as needed

    # Nitro can be used for additional speed, but it's not mentioned in the hints
    # action.nitro = False  # Set to True to use nitro, if desired

    return action


if __name__ == '__main__':
    from utils import PyTux
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